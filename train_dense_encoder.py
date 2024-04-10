#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train DPR Biencoder
"""

import logging
import math
import os
import random
import sys
import time
from typing import Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn

from dpr.models import init_biencoder_components
from dpr.models.biencoder import BiEncoderNllLoss, BiEncoderBatch
from dpr.options import (
    setup_cfg_gpu,
    set_seed,
    get_encoder_params_state_from_cfg,
    set_cfg_params_from_state,
    setup_logger,
)
from dpr.utils.conf_utils import BiencoderDatasetsCfg
from dpr.utils.data_utils import (
    ShardedDataIterator,
    Tensorizer,
    MultiSetDataIterator,
    LocalShardedDataIterator,
)
from dpr.utils.dist_utils import all_gather_list
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    move_to_device,
    get_schedule_linear,
    CheckpointState,
    get_model_file,
    get_model_obj,
    load_states_from_checkpoint,
)

logger = logging.getLogger() # 回一个根日志记录器(root logger)。根日志记录器是日志记录体系中的顶层记录器，如果没有特别指定，所有创建的日志记录器都是它的子记录器
setup_logger(logger)


class BiEncoderTrainer(object):
    """
    这段代码是一个双向编码器（Bi-Encoder）训练器的实现，常用于自然语言处理中的检索任务，比如问答系统。该类包含了训练过程中所需的各种方法，包括初始化、训练循环、验证、保存和加载模型状态等
    """

    def __init__(self, cfg: DictConfig):
        # 根据配置 (cfg) 设置分布式训练的相关参数（如 shard_id 和 distributed_factor）
        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1

        # 通过日志记录器（logger）输出初始化训练组件的信息
        logger.info("***** Initializing components for training *****")

        # 如果指定了模型文件，从该文件加载模型状态，并根据加载的状态调整配置参数
        model_file = get_model_file(cfg, cfg.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_cfg_params_from_state(saved_state.encoder_params, cfg)

        # 初始化双向编码器的主要组件，包括 tensorizer（用于数据的预处理和编码）、model（双向编码器模型）、和 optimizer（优化器）
        tensorizer, model, optimizer = init_biencoder_components(cfg.encoder.encoder_model_type, cfg)

        # 设置模型、优化器为分布式训练模式（如果适用），并处理半精度（FP16）训练的相关设置
        model, optimizer = setup_for_distributed_mode(
            model,
            optimizer,
            cfg.device,
            cfg.n_gpu,
            cfg.local_rank,
            cfg.fp16,
            cfg.fp16_opt_level,
        )
        # 加载保存的状态（如果有），初始化一些控制训练过程的属性，如起始epoch、批次、调度器状态等
        self.biencoder = model
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        self.cfg = cfg
        self.ds_cfg = BiencoderDatasetsCfg(cfg)

        if saved_state:
            self._load_saved_state(saved_state)

        self.dev_iterator = None

    def get_data_iterator(
        self,
        batch_size: int,
        is_train_set: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        rank: int = 0,
    ):
        # 从多个数据集中创建和获取数据迭代器，以供模型训练或验证使用
        """ 
        batch_size: 每个批次的数据大小。
        is_train_set: 一个布尔值，用于指示是否使用训练集（True）或验证集（False）。
        shuffle: 是否在每个epoch开始时打乱数据。
        shuffle_seed: 打乱数据时使用的随机种子，以保证可复现性。
        offset: 从数据集的这个偏移量开始遍历数据。
        rank: 当前进程在分布式训练中的排名。
        """

        # 根据 is_train_set 的值，选择使用训练数据集 (train_datasets) 或验证数据集 (dev_datasets)
        hydra_datasets = self.ds_cfg.train_datasets if is_train_set else self.ds_cfg.dev_datasets
        sampling_rates = self.ds_cfg.sampling_rates

        # 记录初始化的数据集信息
        logger.info(
            "Initializing task/set data %s",
            self.ds_cfg.train_datasets_names if is_train_set else self.ds_cfg.dev_datasets_names,
        )

        # 根据配置 cfg.local_shards_dataloader 的值，选择使用 LocalShardedDataIterator（用于本地分片数据加载）或 ShardedDataIterator（用于分布式数据加载）
        single_ds_iterator_cls = LocalShardedDataIterator if self.cfg.local_shards_dataloader else ShardedDataIterator

        # 对选定的数据集进行遍历，为每个数据集创建一个数据迭代器实例。这里的 shard_id 和 distributed_factor 用于在分布式训练中分配数据的分片，以确保每个训练进程处理数据集的不同部分
        sharded_iterators = [
            single_ds_iterator_cls(
                ds,
                shard_id=self.shard_id,
                num_shards=self.distributed_factor,
                batch_size=batch_size,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
                offset=offset,
            )
            for ds in hydra_datasets
        ]

        """ 
        使用前面创建的分片迭代器列表来初始化 MultiSetDataIterator。这个迭代器可以遍历多个数据集，并根据配置（如是否打乱、采样率等）进行相应的处理
        在训练集上，根据 sampling_rates 来决定每个数据集的采样率。在验证集上，采样率被设置为1（即不进行采样）
        """
        return MultiSetDataIterator(
            sharded_iterators,
            shuffle_seed,
            shuffle,
            sampling_rates=sampling_rates if is_train_set else [1],
            rank=rank,
        )

    def run_train(self):
        # 定义了双向编码器模型的训练逻辑，包括初始化数据迭代器、计算总更新次数、设置学习率调度器，以及执行多个训练周期
        cfg = self.cfg

        # 通过调用 get_data_iterator 方法，根据训练配置（cfg.train），初始化训练数据迭代器。该迭代器用于在每个epoch中遍历训练数据
        train_iterator = self.get_data_iterator(
            cfg.train.batch_size,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=self.start_batch,
            rank=cfg.local_rank,
        )
        max_iterations = train_iterator.get_max_iterations()
        logger.info("  Total iterations per epoch=%d", max_iterations)
        if max_iterations == 0:
            # 如果没有数据可用于训练（max_iterations 为0），则记录警告并终止训练
            logger.warning("No data found for training.")
            return

        # 计算每个epoch需要执行的参数更新次数，这通常与梯度累积的步骤数有关
        updates_per_epoch = train_iterator.max_iterations // cfg.train.gradient_accumulation_steps

        # 整个训练过程中预期执行的总更新次数，基于每个epoch的更新次数和总的训练周期数
        total_updates = updates_per_epoch * cfg.train.num_train_epochs
        logger.info(" Total updates=%d", total_updates)
        warmup_steps = cfg.train.warmup_steps

        if self.scheduler_state:
            # 根据配置设置学习率调度器，这里使用的是线性调度器（get_schedule_linear），它根据训练进度调整学习率

            logger.info("Loading scheduler state %s", self.scheduler_state)
            shift = int(self.scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            scheduler = get_schedule_linear(
                self.optimizer,
                warmup_steps,
                total_updates,
                steps_shift=shift,
            )
        else:
            # 如果存在保存的调度器状态（self.scheduler_state），则从该状态恢复调度器。否则，从头开始新的调度
            scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)

        # 计算每个epoch进行模型验证的步骤数（eval_step），这是基于配置的每个epoch中进行多少次验证（cfg.train.eval_per_epoch）
        eval_step = math.ceil(updates_per_epoch / cfg.train.eval_per_epoch)
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        # 对于从 self.start_epoch 到 cfg.train.num_train_epochs 的每个epoch，执行 _train_epoch 方法，其中包括模型的前向和后向传播、参数更新等
        for epoch in range(self.start_epoch, int(cfg.train.num_train_epochs)):
            logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(scheduler, epoch, eval_step, train_iterator)

        if cfg.local_rank in [-1, 0]:
            # 如果是在非分布式训练环境中或是分布式训练的主进程（cfg.local_rank 为 -1 或 0），在训练完成后，记录最佳验证检查点的名称（self.best_cp_name）
            logger.info("Training finished. Best validation checkpoint %s", self.best_cp_name)

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        # 执行模型的验证过程，并根据验证结果保存检查点
        # 这个方法特别关注于在分布式训练环境中的操作，确保只有特定的进程（通常是主进程）负责保存检查点
        """ 
        epoch: 当前的训练轮次（epoch）。
        iteration: 当前epoch内的迭代次数。
        scheduler: 当前使用的学习率调度器。
        """
        cfg = self.cfg # 获取配置
        # 确定是否保存检查点。在分布式训练中，只有当 cfg.local_rank 为 -1（非分布式）或 0（即主进程）时，才设置为 True
        save_cp = cfg.local_rank in [-1, 0]

        # 如果当前epoch等于配置中指定的开始验证的epoch (cfg.val_av_rank_start_epoch)，则重置最佳验证结果（self.best_validation_result）为 None
        if epoch == cfg.val_av_rank_start_epoch:
            self.best_validation_result = None

        # 检查是否有指定的开发集（dev_datasets）。如果没有，将验证损失设置为0
        if not cfg.dev_datasets:
            validation_loss = 0
        else:
            # 如果当前epoch大于或等于开始验证的epoch，使用 validate_average_rank 方法进行验证；否则，使用 validate_nll 方法。这两个方法对应不同的验证策略，分别用于不同的验证阶段或目的。
            if epoch >= cfg.val_av_rank_start_epoch:
                validation_loss = self.validate_average_rank()
            else:
                validation_loss = self.validate_nll()

        if save_cp:
            # 如果当前进程是保存检查点的进程（save_cp 为 True），则调用 _save_checkpoint 方法保存当前的检查点，并记录检查点名称（cp_name）。
            # 日志记录保存的检查点信息。
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info("Saved checkpoint to %s", cp_name)

            if validation_loss < (self.best_validation_result or validation_loss + 1):
                # 如果当前的验证损失小于之前的最佳验证结果（或者是第一次验证，此时 self.best_validation_result 为 None），则更新最佳验证结果为当前的验证损失，并更新最佳检查点名称为当前检查点的名称
                self.best_validation_result = validation_loss
                self.best_cp_name = cp_name
                logger.info("New Best validation checkpoint %s", cp_name)

    def validate_nll(self) -> float:
        # 验证双向编码器（Bi-Encoder）模型的方法，通过计算负对数似然（NLL）损失来评估模型在开发集（验证集）上的性能
        logger.info("NLL validation ...")
        cfg = self.cfg
        # 将双向编码器模型设置为评估模式（.eval()），这样可以在验证过程中关闭Dropout和Batch Normalization等只在训练时使用的层
        self.biencoder.eval()

        # 如果之前没有创建开发集的数据迭代器（self.dev_iterator），则调用 get_data_iterator 方法创建一个。这个迭代器配置为不打乱数据，使用开发集的批量大小（cfg.train.dev_batch_size）
        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_size, False, shuffle=False, rank=cfg.local_rank
            )
        data_iterator = self.dev_iterator   # 使用创建好的数据迭代器进行数据的迭代

        total_loss = 0.0
        start_time = time.time()
        total_correct_predictions = 0
        num_hard_negatives = cfg.train.hard_negatives
        num_other_negatives = cfg.train.other_negatives
        log_result_step = cfg.train.log_batch_step
        batches = 0
        dataset = 0
        biencoder = get_model_obj(self.biencoder)

        # 对于迭代器中的每个批次，解构批次数据（可能包含数据集的索引，如果是从多个数据集中迭代的话）
        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            logger.info("Eval step: %d ,rnk=%s", i, cfg.local_rank) # 记录当前验证步骤的日志

            # 使用双向编码器的输入创建方法（create_biencoder_input），根据当前批次的样本，准备模型的输入数据。这包括处理硬负样本和其他负样本的数量
            biencoder_input = biencoder.create_biencoder_input(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
            )

            # 根据数据集配置，获取用于表示选择的令牌位置，以及编码器类型
            ds_cfg = self.ds_cfg.dev_datasets[dataset]
            rep_positions = ds_cfg.selector.get_positions(biencoder_input.question_ids, self.tensorizer)
            encoder_type = ds_cfg.encoder_type

            # 执行模型的前向传播，计算损失和正确预测的数量
            loss, correct_cnt = _do_biencoder_fwd_pass(
                self.biencoder,
                biencoder_input,
                self.tensorizer,
                cfg,
                encoder_type=encoder_type,
                rep_positions=rep_positions,
            )
            # 累加每个批次的损失和正确预测的数量。
            total_loss += loss.item()
            total_correct_predictions += correct_cnt
            batches += 1
            # 每隔一定步数（log_result_step），记录当前的验证步骤、使用的时间和损失
            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Eval step: %d , used_time=%f sec., loss=%f ",
                    i,
                    time.time() - start_time,
                    loss.item(),
                )

        total_loss = total_loss / batches # 计算总损失除以批次数得到平均损失
        total_samples = batches * cfg.train.dev_batch_size * self.distributed_factor
        # 计算正确预测的比例
        correct_ratio = float(total_correct_predictions / total_samples)
        # 记录最终的NLL验证损失和正确率
        logger.info(
            "NLL Validation: loss = %f. correct prediction ratio  %d/%d ~  %f",
            total_loss,
            total_correct_predictions,
            total_samples,
            correct_ratio,
        )
        return total_loss

    def validate_average_rank(self) -> float:
        """
        使用平均排名（Average Rank）作为评估指标的验证方法，用于验证双向编码器（Bi-Encoder）模型。这种方法主要应用于信息检索任务，比如问答系统中，评估模型检索相关信息的能力。具体来说，这个方法通过计算每个问题的黄金（正确）通道在所有通道集中的平均排名来进行模型的验证
        目的：使用每个问题的黄金通道的排名来验证双向编码器模型的性能。
        过程：为每个问题生成表示向量和指定数量的负样本表示向量，然后计算所有问题向量与通道向量之间的相似度得分，并对每个问题按得分进行排序。最后，计算所有问题的黄金通道的平均排名。
        """
        logger.info("Average rank validation ...")

        cfg = self.cfg
        self.biencoder.eval() # 将模型设置为评似模式，以禁用训练阶段特有的操作，如 Dropout
        distributed_factor = self.distributed_factor

        # 如果之前未创建开发集的数据迭代器，则创建一个新的迭代器用于验证
        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                cfg.train.dev_batch_size, False, shuffle=False, rank=cfg.local_rank
            )
        data_iterator = self.dev_iterator

        """ 
        sub_batch_size、num_hard_negatives、num_other_negatives、log_result_step 从配置对象 cfg 中获取，分别表示子批次大小、硬负样本数、其他负样本数和日志记录步骤。
        sim_score_f 是一个计算相似度分数的函数，可能用于计算问题和上下文表示向量之间的相似度。
        q_represenations 和 ctx_represenations 分别用于存储问题和上下文的表示向量。
        positive_idx_per_question 用于记录每个问题对应的正样本索引。
        """
        sub_batch_size = cfg.train.val_av_rank_bsz
        sim_score_f = BiEncoderNllLoss.get_similarity_function()
        q_represenations = []
        ctx_represenations = []
        positive_idx_per_question = []

        num_hard_negatives = cfg.train.val_av_rank_hard_neg
        num_other_negatives = cfg.train.val_av_rank_other_neg

        log_result_step = cfg.train.log_batch_step
        dataset = 0
        biencoder = get_model_obj(self.biencoder)
        # 通过遍历数据迭代器，为每个批次的问题和通道生成表示向量。考虑了硬负样本和其他负样本的数量
        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            # samples += 1
            # 检查问题表示的数量是否已经达到了配置中指定的最大值(cfg.train.val_av_rank_max_qs)，考虑到分布式因子(distributed_factor)。如果是，循环会提前结束
            if len(q_represenations) > cfg.train.val_av_rank_max_qs / distributed_factor:
                break

            # 检查samples_batch是否是一个元组（Tuple），这意味着批次数据可能还包含其他信息（比如数据集索引）。如果是，将samples_batch和dataset分别赋值
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            # 使用 biencoder.create_biencoder_input 方法创建双编码器的输入，该方法根据批次数据、tensorizer（负责数据张量化）、硬负样本和其他负样本的数量等参数生成输入
            biencoder_input = biencoder.create_biencoder_input(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
            )
            total_ctxs = len(ctx_represenations)
            # ctxs_ids 和 ctxs_segments 分别存储上下文的ID和段信息，bsz 是当前批次的大小。
            ctxs_ids = biencoder_input.context_ids
            ctxs_segments = biencoder_input.ctx_segments
            bsz = ctxs_ids.size(0)

            # 通过数据集配置(ds_cfg)获取编码器类型(encoder_type)和选择器(selector)
            ds_cfg = self.ds_cfg.dev_datasets[dataset]
            encoder_type = ds_cfg.encoder_type
            # 使用选择器的 get_positions 方法和提供的问题ID(biencoder_input.question_ids)以及tensorizer来确定用于表示选择的令牌位置
            rep_positions = ds_cfg.selector.get_positions(biencoder_input.question_ids, self.tensorizer)

            # 使用for循环和enumerate函数按照子批次大小（sub_batch_size）遍历整个批次的数据（bsz）
            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):

                # 在每个子批次的第一组数据中，获取问题的ID(q_ids)和段落ID(q_segments)。对于后续的子批次，这些值被设置为None，因为问题信息只在第一个子批次中处理
                q_ids, q_segments = (
                    (biencoder_input.question_ids, biencoder_input.question_segments) if j == 0 else (None, None)
                )

                if j == 0 and cfg.n_gpu > 1 and q_ids.size(0) == 1:
                    # 如果是在数据并行（DP）模式下运行，而不是分布式数据并行（DDP）模式，并且如果问题ID的大小为1，此时跳过当前迭代。这是因为在DP模式下，所有模型输入张量的批次大小应该大于1或等于0，否则在分割输入张量时可能会出现问题
                    continue

                # 从ctxs_ids和ctxs_segments中提取当前子批次对应的上下文ID和段落ID
                ctx_ids_batch = ctxs_ids[batch_start : batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[batch_start : batch_start + sub_batch_size]

                # 使用self.tensorizer.get_attn_mask方法为问题ID和上下文ID生成注意力掩码（q_attn_mask和ctx_attn_mask）。注意力掩码用于在模型的自注意力层中标识有效的输入位置
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch)
                # 在torch.no_grad()上下文中执行模型的前向传播，以避免计算梯度并减少内存使用。此步骤生成问题和上下文的密集表示向量（q_dense和ctx_dense）
                with torch.no_grad():
                    q_dense, ctx_dense = self.biencoder(
                        q_ids,
                        q_segments,
                        q_attn_mask,
                        ctx_ids_batch,
                        ctx_seg_batch,
                        ctx_attn_mask,
                        encoder_type=encoder_type,
                        representation_token_pos=rep_positions,
                    )

                # 如果问题的密集表示（q_dense）不为None（即在第一个子批次中处理过），则将其添加到q_represenations列表中。同时，将上下文的密集表示向量添加到ctx_represenations列表中
                if q_dense is not None:
                    q_represenations.extend(q_dense.cpu().split(1, dim=0))

                ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            # 从biencoder_input中获取标记为正样本的索引，并将这些索引（考虑到已处理的上下文数量total_ctxs）添加到positive_idx_per_question列表中，用于后续计算模型的平均排名
            batch_positive_idxs = biencoder_input.is_positive
            positive_idx_per_question.extend([total_ctxs + v for v in batch_positive_idxs])

            # 每当处理的数据批次数达到日志记录步骤（log_result_step）时，记录当前计算的上下文向量和问题向量的数量
            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Av.rank validation: step %d, computed ctx_vectors %d, q_vectors %d",
                    i,
                    len(ctx_represenations),
                    len(q_represenations),
                )

        # 使用torch.cat将之前收集的问题和上下文表示向量列表合并成两个大的张量（ctx_represenations和q_represenations）
        ctx_represenations = torch.cat(ctx_represenations, dim=0)
        q_represenations = torch.cat(q_represenations, dim=0)

        # 记录问题向量和上下文向量的总数，以便了解验证集的规模和模型处理的数据量
        logger.info("Av.rank validation: total q_vectors size=%s", q_represenations.size())
        logger.info("Av.rank validation: total ctx_vectors size=%s", ctx_represenations.size())

        # 确认问题向量的数量（q_num）与正样本索引列表的长度相等，这是一个基本的一致性检查
        q_num = q_represenations.size(0)
        assert q_num == len(positive_idx_per_question)

        # 使用之前获取的相似度函数（sim_score_f）计算问题向量和上下文向量之间的相似度分数
        scores = sim_score_f(q_represenations, ctx_represenations)
        values, indices = torch.sort(scores, dim=1, descending=True)

        # 对每个问题，根据其相似度分数对上下文进行排序（降序），然后找出正确上下文的索引在排序中的位置（即排名）
        rank = 0
        for i, idx in enumerate(positive_idx_per_question):
            # aggregate the rank of the known gold passage in the sorted results for each question
            gold_idx = (indices[i] == idx).nonzero()
            # 遍历所有问题，累积每个问题正确上下文的排名，以计算总排名
            rank += gold_idx.item()

        if distributed_factor > 1:
            # 如果使用了分布式训练（distributed_factor > 1），则通过在各个计算节点间交换排名信息来计算全局平均排名。这确保了即使在分布式设置下，也能得到整体的性能评估
            eval_stats = all_gather_list([rank, q_num], max_size=100)
            for i, item in enumerate(eval_stats):
                remote_rank, remote_q_num = item
                if i != cfg.local_rank:
                    rank += remote_rank
                    q_num += remote_q_num

        # 通过总排名除以问题数量来计算平均排名，并记录下来。平均排名反映了模型在定位正确上下文方面的整体性能
        av_rank = float(rank / q_num)
        logger.info("Av.rank validation: average rank %s, total questions=%d", av_rank, q_num)
        # 双编码器模型的验证阶段，如何通过计算平均排名来评估模型性能。平均排名越低，意味着模型在将正确答案排在前面的能力越强，这在问答系统或信息检索等任务中是非常重要的性能指标
        return av_rank

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        eval_step: int,
        train_data_iterator: MultiSetDataIterator,
    ):
        # 执行单个epoch训练的实现。这个方法涉及数据的处理、模型的前向和后向传播、梯度的累积与优化、以及定期的性能验证和日志记录

        cfg = self.cfg
        rolling_train_loss = 0.0
        # 初始化用于累积的变量，如epoch_loss和epoch_correct_predictions，分别用于跟踪损失和正确预测的数量
        epoch_loss = 0
        epoch_correct_predictions = 0

        # 从配置(cfg)中提取相关参数，如日志记录步骤(log_result_step)、损失缩放因子(loss_scale)等
        log_result_step = cfg.train.log_batch_step
        rolling_loss_step = cfg.train.train_rolling_loss_step
        num_hard_negatives = cfg.train.hard_negatives
        num_other_negatives = cfg.train.other_negatives
        seed = cfg.seed
        self.biencoder.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0

        biencoder = get_model_obj(self.biencoder)
        dataset = 0
        # 使用train_data_iterator遍历训练数据。train_data_iterator负责提供数据批次，可能包括对数据的预处理
        for i, samples_batch in enumerate(train_data_iterator.iterate_ds_data(epoch=epoch)):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            ds_cfg = self.ds_cfg.train_datasets[dataset]
            special_token = ds_cfg.special_token
            encoder_type = ds_cfg.encoder_type
            shuffle_positives = ds_cfg.shuffle_positives

            # 对于每个数据批次，根据需要处理数据（比如随机种子的设置、数据混洗）
            data_iteration = train_data_iterator.get_iteration()
            random.seed(seed + epoch + data_iteration)

            # 创建模型的输入（biencoder_batch），包括处理硬负样本、其他负样本、特殊令牌等
            biencoder_batch = biencoder.create_biencoder_input(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=True,
                shuffle_positives=shuffle_positives,
                query_token=special_token,
            )

            # get the token to be used for representation selection
            from dpr.utils.data_utils import DEFAULT_SELECTOR

            selector = ds_cfg.selector if ds_cfg else DEFAULT_SELECTOR

            rep_positions = selector.get_positions(biencoder_batch.question_ids, self.tensorizer)

            loss_scale = cfg.loss_scale_factors[dataset] if cfg.loss_scale_factors else None
            # 执行模型的前向传播（_do_biencoder_fwd_pass），计算损失和正确预测的数量
            loss, correct_cnt = _do_biencoder_fwd_pass(
                self.biencoder,
                biencoder_batch,
                self.tensorizer,
                cfg,
                encoder_type=encoder_type,
                rep_positions=rep_positions,
                loss_scale=loss_scale,
            )

            epoch_correct_predictions += correct_cnt
            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            # 如果配置(cfg)启用了FP16训练，使用apex.amp来处理损失的反向传播。amp.scale_loss函数用于缩放损失以防止FP16训练中的数值下溢。接着，如果配置了最大梯度范数(max_grad_norm)，使用torch.nn.utils.clip_grad_norm_来裁剪amp管理的优化器参数的梯度，防止梯度爆炸
            if cfg.fp16:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), cfg.train.max_grad_norm)
            # 在没有启用FP16的情况下，直接对损失调用.backward()进行反向传播，然后可能进行梯度裁剪
            else:
                loss.backward()
                if cfg.train.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.biencoder.parameters(), cfg.train.max_grad_norm)

            # 考虑到梯度累积设置(gradient_accumulation_steps)，只有当当前步骤是累积步骤的倍数时，才执行优化器的step方法来更新模型参数，并执行学习率调度器的step方法来更新学习率，然后清零模型的梯度
            if (i + 1) % cfg.train.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.biencoder.zero_grad()

            # 在每个log_result_step步骤，记录当前的学习率和损失值，以便监控训练过程
            if i % log_result_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch: %d: Step: %d/%d, loss=%f, lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    loss.item(),
                    lr,
                )

            # 在每个rolling_loss_step步骤，计算并记录最近一段时间内的平均损失，然后重置滚动损失计数器
            if (i + 1) % rolling_loss_step == 0:
                logger.info("Train batch %d", data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info(
                    "Avg. loss per last %d batches: %f",
                    rolling_loss_step,
                    latest_rolling_train_av_loss,
                )
                rolling_train_loss = 0.0

            # 在每个eval_step步骤，执行模型的验证过程，并保存当前的模型状态。这一步骤是检查模型性能并进行模型选择的关键环节
            if data_iteration % eval_step == 0:
                logger.info(
                    "rank=%d, Validation: Epoch: %d Step: %d/%d",
                    cfg.local_rank,
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(epoch, train_data_iterator.get_iteration(), scheduler)
                self.biencoder.train()

        # 记录当前节点（在分布式训练中）完成了一个训练epoch的信息
        logger.info("Epoch finished on %d", cfg.local_rank)
        # 在每个epoch结束时调用validate_and_save方法进行性能验证，并根据验证结果决定是否保存当前模型。这一步是模型训练周期的重要组成部分，它不仅提供了模型性能的即时反馈，还能通过保存模型检查点来防止意外中断导致的进度丢失
        self.validate_and_save(epoch, data_iteration, scheduler)

        # 计算整个epoch的平均损失。这是通过将累积的损失除以处理的批次数来实现的，它提供了一个衡量模型在整个训练周期内性能的指标
        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        # 记录整个epoch的平均损失。这有助于跟踪模型训练过程中的进展，并对模型的学习率等超参数做出调整
        logger.info("Av Loss per epoch=%f", epoch_loss)
        # 记录整个epoch中正确预测的总数。这个数字提供了一个直观的性能指标，显示了模型在训练数据上的准确度
        logger.info("epoch total correct predictions=%d", epoch_correct_predictions)

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        # 在训练过程中保存当前模型的状态，包括模型参数、优化器状态、学习率调度器状态以及当前的epoch和batch offset
        """ 
        scheduler: 学习率调度器对象，用于调整训练过程中的学习率。
        epoch: 当前的epoch编号。
        offset: 当前epoch内已处理的batch数量
        """
        cfg = self.cfg # 获取训练配置
        model_to_save = get_model_obj(self.biencoder) # 通过调用get_model_obj函数获取要保存的模型对象。如果模型使用了如DataParallel或DistributedDataParallel的封装，该函数确保保存的是原始模型参数而非封装后的
        # 构造检查点文件的路径。检查点文件名由配置中的输出目录、检查点文件基础名和当前epoch编号组成，从而方便标识和管理不同时间点的检查点文件
        cp = os.path.join(cfg.output_dir, cfg.checkpoint_file_name + "." + str(epoch))
        # 获取根据配置文件定义的额外模型参数。这通常包括模型特定的配置项，如编码器的参数
        meta_params = get_encoder_params_state_from_cfg(cfg)
        # 创建一个CheckpointState对象，该对象包含模型的状态字典、优化器的状态字典、调度器的状态字典、当前的offset、epoch和额外的模型参数
        state = CheckpointState(
            model_to_save.get_state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        # 使用PyTorch的torch.save方法保存检查点状态
        # state._asdict()将CheckpointState对象转换为字典，cp是文件保存路径
        torch.save(state._asdict(), cp)
        # 使用日志记录器记录检查点的保存路径
        logger.info("Saved checkpoint at %s", cp)
        return cp # 返回保存的检查点文件的路径

    def _load_saved_state(self, saved_state: CheckpointState):
        # 从一个保存的状态（CheckpointState对象）中加载模型的状态，可能还包括优化器和学习率调度器的状态
        epoch = saved_state.epoch # 获取保存的状态中的epoch值
        # 获取保存的状态中的offset值。这里的offset通常指的是在当前epoch内已完成的批次(batch)数目。但这个值在这个方法中被忽略了，因为所有检查点都是在完成完整的epoch之后创建的
        offset = saved_state.offset
        if offset == 0: 
            # 如果offset等于0，意味着一个epoch已经完全完成了，因此epoch值加1
            epoch += 1
        # 记录当前正在加载的检查点的batch数和epoch数
        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)

        # 根据配置（self.cfg.ignore_checkpoint_offset），决定是从头开始（即start_epoch=0, start_batch=0），还是从保存的epoch和batch开始继续训练
        if self.cfg.ignore_checkpoint_offset:
            self.start_epoch = 0
            self.start_batch = 0
        else:
            self.start_epoch = epoch
            # TODO: offset doesn't work for multiset currently
            self.start_batch = 0  # offset

        model_to_load = get_model_obj(self.biencoder) # 获取实际需要加载状态的模型对象
        logger.info("Loading saved model state ...")

        model_to_load.load_state(saved_state, strict=True) # 加载保存的模型状态。这里的strict=True参数意味着加载时会确保模型的参数与保存状态完全匹配，任何不匹配都会引发错误。
        logger.info("Saved state loaded")
        if not self.cfg.ignore_checkpoint_optimizer:
            if saved_state.optimizer_dict:
                # 如果配置没有指定忽略优化器状态（ignore_checkpoint_optimizer为False），且保存的状态中包含优化器的状态字典（optimizer_dict），那么就使用这个保存的优化器状态
                logger.info("Using saved optimizer state")
                self.optimizer.load_state_dict(saved_state.optimizer_dict)

        # 如果配置没有指定忽略学习率调度器的状态（ignore_checkpoint_lr为False），且保存的状态中包含学习率调度器的状态字典（scheduler_dict），那么就使用这个保存的状态
        if not self.cfg.ignore_checkpoint_lr and saved_state.scheduler_dict:
            logger.info("Using saved scheduler_state")
            self.scheduler_state = saved_state.scheduler_dict


def _calc_loss(
    cfg,
    loss_function,
    local_q_vector,
    local_ctx_vectors,
    local_positive_idxs,
    local_hard_negatives_idxs: list = None,
    loss_scale: float = None,
) -> Tuple[T, bool]:
    """
    这个 _calc_loss 函数是一个专门用于计算损失的方法，支持在分布式数据并行（DDP）模式下运行，通过在所有节点之间交换表示（向量），实现在批次内负样本方案的损失计算。它旨在处理那些需要跨多个处理单元共享数据以计算损失的情况，特别是在训练大规模模型时非常有用
    """
    distributed_world_size = cfg.distributed_world_size or 1
    # 如果配置中指定了分布式世界大小（distributed_world_size）大于1，表示在分布式训练模式下运行。首先，问题向量（local_q_vector）和上下文向量（local_ctx_vectors）被复制到CPU并脱离计算图（为了跨节点交换）
    if distributed_world_size > 1:
        q_vector_to_send = torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        ctx_vector_to_send = torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()

        # 使用 all_gather_list 函数将每个节点上的这些向量及其对应的正样本索引和硬负样本索引收集起来，形成全局数据集
        global_question_ctx_vectors = all_gather_list(
            [
                q_vector_to_send,
                ctx_vector_to_send,
                local_positive_idxs,
                local_hard_negatives_idxs,
            ],
            max_size=cfg.global_loss_buf_sz,
        )

        global_q_vector = []
        global_ctxs_vector = []

        # ctxs_per_question = local_ctx_vectors.size(0)
        positive_idx_per_question = []
        hard_negatives_per_question = []

        total_ctxs = 0

        # 这些全局收集的数据被合并到单个的问题向量和上下文向量中，并更新正样本和硬负样本的索引以反映全局的上下文索引
        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx, hard_negatives_idxs = item

            # 在非分布式训练模式下，直接使用本地问题向量、上下文向量及其对应的索引来计算损失。在分布式训练模式下，使用合并后的全局问题向量、全局上下文向量及更新后的索引来计算损失
            if i != cfg.local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in hard_negatives_idxs])
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                positive_idx_per_question.extend([v + total_ctxs for v in local_positive_idxs])
                hard_negatives_per_question.extend([[v + total_ctxs for v in l] for l in local_hard_negatives_idxs])
            total_ctxs += ctx_vectors.size(0)
        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negatives_idxs

    # 通过调用传入的损失函数loss_function.calc，传入问题向量、上下文向量、正样本索引和硬负样本索引（以及可选的损失缩放因子loss_scale），计算损失和正确性指标（是否正确）
    loss, is_correct = loss_function.calc(
        global_q_vector,
        global_ctxs_vector,
        positive_idx_per_question,
        hard_negatives_per_question,
        loss_scale=loss_scale,
    )

    return loss, is_correct # 返回计算得到的损失和正确性指标（is_correct）


def _print_norms(model):
    # 计算并返回一个模型中所有梯度的L2范数（即欧几里得范数）的总和
    total_norm = 0  # 初始化 total_norm 变量为0。这个变量将用于累加所有参数的梯度的L2范数的平方
    # model.named_parameters() 返回一个迭代器，包含模型中每个参数的名称 (n) 和参数对象 (p)。这里的 p 包含了参数的值、梯度等信息
    for n, p in model.named_parameters():
        # 检查参数的梯度是否存在。在某些情况下，如果参数没有参与到前向传播中，它的梯度可能是 None。如果是这种情况，代码将跳过当前的参数，继续下一个
        if p.grad is None:
            continue
        # 计算当前参数的梯度的L2范数。p.grad 获取参数的梯度，data 属性访问梯度的数据，norm(2) 函数计算L2范数
        param_norm = p.grad.data.norm(2)
        # 将当前参数的梯度的L2范数的平方加到 total_norm 变量上。param_norm.item() 将 param_norm 从一个单元素张量转换成一个Python数值，然后平方
        total_norm += param_norm.item() ** 2
    # 在遍历完所有参数并累加它们的梯度的L2范数的平方后，计算平方根，得到所有梯度的L2范数的总和
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def _do_biencoder_fwd_pass(
    model: nn.Module,
    input: BiEncoderBatch,
    tensorizer: Tensorizer,
    cfg,
    encoder_type: str,
    rep_positions=0,
    loss_scale: float = None,
) -> Tuple[torch.Tensor, int]:
    # 封装了双编码器模型的一次前向传播过程，包括损失的计算和正确性检查

    # 将输入批次数据(BiEncoderBatch)移动到指定的设备上(cfg.device)，以确保模型和数据在相同的设备上进行计算
    input = BiEncoderBatch(**move_to_device(input._asdict(), cfg.device))

    # 使用tensorizer.get_attn_mask为问题ID和上下文ID计算注意力掩码。注意力掩码用于指示模型在计算自注意力时应忽略哪些位置，通常用于处理变长输入或padding
    q_attn_mask = tensorizer.get_attn_mask(input.question_ids)
    ctx_attn_mask = tensorizer.get_attn_mask(input.context_ids)

    # 根据模型是否处于训练模式，使用torch.no_grad()上下文管理器来控制是否需要计算梯度。调用模型的前向方法，传入问题ID、问题段落ID、问题的注意力掩码、上下文ID、上下文段落ID、上下文的注意力掩码、编码器类型和代表性令牌位置
    if model.training:
        model_out = model(
            input.question_ids,
            input.question_segments,
            q_attn_mask,
            input.context_ids,
            input.ctx_segments,
            ctx_attn_mask,
            encoder_type=encoder_type,
            representation_token_pos=rep_positions,
        )
    else:
        with torch.no_grad():
            model_out = model(
                input.question_ids,
                input.question_segments,
                q_attn_mask,
                input.context_ids,
                input.ctx_segments,
                ctx_attn_mask,
                encoder_type=encoder_type,
                representation_token_pos=rep_positions,
            )

    local_q_vector, local_ctx_vectors = model_out

    loss_function = BiEncoderNllLoss()  # 初始化用于计算损失的BiEncoderNllLoss对象

    # 调用_calc_loss函数，传入配置、损失函数、问题向量、上下文向量、正样本标志、硬负样本索引和损失缩放因子，以计算损失和正确性指标。这里，is_correct返回的是每个样本的正确性，使用.sum().item()来获取总的正确预测数
    loss, is_correct = _calc_loss(
        cfg,
        loss_function,
        local_q_vector,
        local_ctx_vectors,
        input.is_positive,
        input.hard_negatives,
        loss_scale=loss_scale,
    )
    is_correct = is_correct.sum().item()

    # 如果配置为多GPU训练(cfg.n_gpu > 1)，则对损失求平均，以便在不同GPU间同步损失值。
    if cfg.n_gpu > 1:
        loss = loss.mean()
    # 如果配置了梯度累积(cfg.train.gradient_accumulation_steps > 1)，则相应地调整损失值，这是在实现梯度累积策略时的一个常见做法，用于处理大批量大小的情况
    if cfg.train.gradient_accumulation_steps > 1:
        loss = loss / cfg.train.gradient_accumulation_steps
    return loss, is_correct # 返回计算得到的损失和整个批次中正确预测的总数

# Hydra是一个用于简化配置管理的Python框架，它允许开发者通过配置文件来管理应用程序的参数，从而使代码更加模块化和易于维护
""" 
@hydra.main(config_path="conf", config_name="biencoder_train_cfg")是一个装饰器，用于标记主函数。它告诉Hydra在启动应用程序时加载配置文件
config_path="conf": 指定配置文件的路径。这里假设conf目录包含了应用程序的所有配置文件，这个目录应该位于与启动脚本相同的级别或在指定的路径中。
config_name="biencoder_train_cfg": 指定主配置文件的名称。Hydra会从config_path指定的目录中加载这个名为biencoder_train_cfg的配置文件。这个文件定义了应用程序的主要配置，如模型参数、训练参数等。
"""
@hydra.main(config_path="conf", config_name="biencoder_train_cfg")
# 深度学习训练流程的主函数（main），使用配置对象 cfg 来初始化和运行训练或验证过程。它首先进行一系列的配置检查和设置，然后根据配置决定执行训练还是验证
def main(cfg: DictConfig):
    # 检查配置中的 gradient_accumulation_steps 是否小于1。梯度累积是一种技术，用于在更新模型参数之前，累积多个小批量的梯度，这个参数指定了累积的步骤数
    if cfg.train.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                cfg.train.gradient_accumulation_steps
            )
        )

    if cfg.output_dir is not None:
        # 如果指定了输出目录，则创建该目录。如果目录已经存在，exist_ok=True 参数避免抛出异常
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = setup_cfg_gpu(cfg)    # 根据配置调整 GPU 设置
    set_seed(cfg)   # 设置随机种子，确保实验的可重复性

    if cfg.local_rank in [-1, 0]:
        # 确定是否需要打印配置信息。在分布式训练中，local_rank 表示当前进程的局部排名，只有当 local_rank 为 -1 或 0 时，才打印配置信息
        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

    trainer = BiEncoderTrainer(cfg) # 创建一个 BiEncoderTrainer 实例，用于训练或验证。BiEncoderTrainer 是一个假设存在的类，负责实际的训练或验证流程

    # 检查是否指定了训练数据集，且数据集非空
    if cfg.train_datasets and len(cfg.train_datasets) > 0:
        trainer.run_train()
    # 如果没有指定训练数据集，但指定了模型文件和验证数据集，则执行两种类型的验证
    elif cfg.model_file and cfg.dev_datasets:
        logger.info("No train files are specified. Run 2 types of validation for specified model file")
        trainer.validate_nll()
        trainer.validate_average_rank()
    else:
        #  如果既没有指定训练数据集，也没有同时指定模型文件和验证数据集，则打印一条警告信息，表示没有足够的参数来执行任何操作
        logger.warning("Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do.")


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)   # 记录原始的命令行参数（sys.argv），这有助于调试和记录实验的配置
    hydra_formatted_args = []   # 存放转换后符合Hydra格式的命令行参数
    # 通过遍历sys.argv中的每个参数，检查参数是否以"--"开头。对于这些参数，移除"--"并将剩余部分添加到hydra_formatted_args列表中；对于不以"--"开头的参数，直接添加到列表中。这一步骤是必要的，因为Hydra期望命令行参数不带"--"前缀，而torch.distributed.launch添加的参数通常带有这个前缀
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    # 使用logger.info记录转换后的命令行参数（hydra_formatted_args），这有助于验证参数格式是否已正确调整
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    # 将sys.argv更新为hydra_formatted_args，以确保当Hydra解析命令行参数时，它们是按照Hydra的期望格式提供的
    sys.argv = hydra_formatted_args

    main()
