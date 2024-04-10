#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train the reader model on top of the retriever results
"""

import collections
import json
import sys

import hydra
import logging
import numpy as np
import os
import torch

from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
from typing import List


from dpr.data.qa_validation import exact_match_score
from dpr.data.reader_data import (
    ReaderSample,
    get_best_spans,
    SpanPrediction,
    ExtractiveReaderDataset,
)
from dpr.models import init_reader_components
from dpr.models.reader import create_reader_input, ReaderBatch, compute_loss
from dpr.options import (
    setup_cfg_gpu,
    set_seed,
    set_cfg_params_from_state,
    get_encoder_params_state_from_cfg,
    setup_logger,
)
from dpr.utils.data_utils import (
    ShardedDataIterator,
)
from dpr.utils.model_utils import (
    get_schedule_linear,
    load_states_from_checkpoint,
    move_to_device,
    CheckpointState,
    get_model_file,
    setup_for_distributed_mode,
    get_model_obj,
)

logger = logging.getLogger()
setup_logger(logger)

ReaderQuestionPredictions = collections.namedtuple("ReaderQuestionPredictions", ["id", "predictions", "gold_answers"])


class ReaderTrainer(object):
    # 训练一个阅读理解模型。该类涵盖了模型的初始化、数据准备、训练、验证以及保存检查点等功能
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1 # 参与计算的总进程数

        logger.info("***** Initializing components for training *****")

        # 根据配置中的 checkpoint_file_name 获取模型文件路径，并将结果保存在 model_file 变量中
        model_file = get_model_file(self.cfg, self.cfg.checkpoint_file_name)
        # 存储从检查点中加载的状
        saved_state = None
        if model_file:
            # 执行模型文件的加载和配置更新
            logger.info("!! model_file = %s", model_file)
            saved_state = load_states_from_checkpoint(model_file)
            set_cfg_params_from_state(saved_state.encoder_params, cfg)

        # 根据配置初始化阅读器组件，包括 tensorizer、reader 和 optimizer，并将它们分别保存到对应的变量中
        tensorizer, reader, optimizer = init_reader_components(cfg.encoder.encoder_model_type, cfg)
        # 对 reader 和 optimizer 进行设置，以支持分布式模式。这包括设备、GPU数量、本地排名、FP16精度等的配置
        reader, optimizer = setup_for_distributed_mode(
            reader,
            optimizer,
            cfg.device,
            cfg.n_gpu,
            cfg.local_rank,
            cfg.fp16,
            cfg.fp16_opt_level,
        )
        # 将初始化的 reader、optimizer 和 tensorizer 保存为类的成员变量
        self.reader = reader
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        # 初始化两个变量 start_epoch 和 start_batch，分别用于跟踪训练的起始周期和批次，通常用于恢复训练
        self.start_epoch = 0
        self.start_batch = 0
        # 初始化几个用于跟踪训练进度和性能的变量，如调度器状态、最佳验证结果和最佳检查点名称
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        if saved_state:
            # 如果存在从检查点加载的状态，则调用 _load_saved_state 方法来恢复这些状态
            self._load_saved_state(saved_state)

    def get_data_iterator(
        self,
        path: str,
        batch_size: int,
        is_train: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
    ) -> ShardedDataIterator:
        """ 
        创建和返回一个数据迭代器，该迭代器能够在模型训练和验证时遍历数据集
        
        path: 数据文件的路径。
        batch_size: 每个批次的样本数量。
        is_train: 指示迭代器是用于训练还是验证的布尔值。
        shuffle: 是否在每个纪元开始时随机打乱数据。
        shuffle_seed: 打乱数据时使用的随机种子，确保结果可复现。
        offset: 从数据集的哪个位置开始迭代数据。
        """
        # 根据当前的分布式训练设置决定是否运行数据预处理。如果是单机训练或当前进程是主进程（local_rank 为 -1 或 0），则 run_preprocessing 为 True
        run_preprocessing = True if self.distributed_factor == 1 or self.cfg.local_rank in [-1, 0] else False

        # gold_passages_src 是包含标准答案的数据文件的路径。如果这个路径存在且当前不是在训练模式下，方法会尝试使用验证集的黄金通道源（gold_passages_src_dev）。这里通过 assert 语句确保了指定的路径有效
        gold_passages_src = self.cfg.gold_passages_src
        if gold_passages_src:
            if not is_train:
                gold_passages_src = self.cfg.gold_passages_src_dev

            assert os.path.exists(gold_passages_src), "Please specify valid gold_passages_src/gold_passages_src_dev"

        # 使用 ExtractiveReaderDataset 类来加载和处理数据。这个类需要数据路径、是否为训练模式、黄金通道源路径、文本处理器（tensorizer）、是否运行预处理，以及工作进程数作为输入
        dataset = ExtractiveReaderDataset(
            path,
            is_train,
            gold_passages_src,
            self.tensorizer,
            run_preprocessing,
            self.cfg.num_workers,
        )

        dataset.load_data() # 加载数据集

        # 创建一个数据迭代器。这个迭代器负责管理数据的分片（对于分布式训练来说很重要）、批次大小、是否随机打乱以及其他设置
        iterator = ShardedDataIterator(
            dataset,
            shard_id=self.shard_id,
            num_shards=self.distributed_factor,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            offset=offset,
        )
        iterator.calculate_shards() # 计算每个分片的数据量

        # apply deserialization hook
        # 应用一个反序列化钩子，对每个样本进行处理。这个步骤通常用于将数据从其存储格式转换为模型训练或验证时需要的格式
        iterator.apply(lambda sample: sample.on_deserialize())
        return iterator

    def run_train(self):
        # 管理模型的整个训练过程
        cfg = self.cfg # 获取训练配置

        # 创建一个训练数据迭代器 train_iterator，配置如批大小、是否打乱、种子等
        train_iterator = self.get_data_iterator(
            cfg.train_files,
            cfg.train.batch_size,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=self.start_batch,
        )

        # num_train_epochs = cfg.train.num_train_epochs - self.start_epoch
        # 打印日志以显示每个纪元的总迭代次数和整个训练过程中的总更新次数
        logger.info("Total iterations per epoch=%d", train_iterator.max_iterations)
        updates_per_epoch = train_iterator.max_iterations // cfg.train.gradient_accumulation_steps

        total_updates = updates_per_epoch * cfg.train.num_train_epochs
        logger.info(" Total updates=%d", total_updates)

        warmup_steps = cfg.train.warmup_steps

        # 根据配置中的 warmup_steps 和总更新次数初始化学习率调度器 scheduler。如果有保存的调度器状态，则从该状态恢复
        if self.scheduler_state:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            shift = int(self.scheduler_state["last_epoch"])
            logger.info("Steps shift %d", shift)
            scheduler = get_schedule_linear(
                self.optimizer,
                warmup_steps,
                total_updates,
            )
        else:
            scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)

        # 设定全局步数 global_step，它结合了开始的纪元数和更新次数来确保训练可以从中断处恢复
        eval_step = cfg.train.eval_step
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        global_step = self.start_epoch * updates_per_epoch + self.start_batch

        for epoch in range(self.start_epoch, cfg.train.num_train_epochs):
            # 对每个训练纪元，调用 _train_epoch 方法执行一个完整的训练周期。这包括数据迭代、损失计算、反向传播和权重更新。
            # 在每个纪元的训练过程中，根据配置的评估步骤 eval_step，进行模型验证并可能保存当前的最佳检查点
            logger.info("***** Epoch %d *****", epoch)
            global_step = self._train_epoch(scheduler, epoch, eval_step, train_iterator, global_step)

        if cfg.local_rank in [-1, 0]:
            # 训练完成后，如果当前进程为主进程（local_rank 为 -1 或 0），则记录训练完成的日志信息，包括最佳验证检查点的名称
            logger.info("Training finished. Best validation checkpoint %s", self.best_cp_name)

        return

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        """ 
        在每个训练周期后执行模型验证，并在必要时保存当前的模型检查点
        
        epoch: 当前的训练纪元。
        iteration: 当前纪元中的迭代次数。
        scheduler: 当前使用的学习率调度器。
        """
        # 从实例变量 cfg 中获取配置，这通常包含了训练、验证和模型配置信息
        cfg = self.cfg
        # in distributed DDP mode, save checkpoint for only one process
        # 在分布式数据并行（DDP）模式下，只有主进程（local_rank 为 -1 或 0）会保存检查点，以避免多个进程重复写入相同的文件
        save_cp = cfg.local_rank in [-1, 0]
        # 调用 validate 方法执行模型验证，并返回验证分数（reader_validation_score）。这个分数通常用于评估模型在验证集上的性能
        reader_validation_score = self.validate()

        if save_cp:
            # 如果当前进程是主进程（即允许保存检查点），则调用 _save_checkpoint 方法保存当前模型状态的检查点。检查点的名称（cp_name）包含了纪元和迭代次数的信息，以确保唯一性
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info("Saved checkpoint to %s", cp_name)

            if reader_validation_score < (self.best_validation_result or 0):
                # 检查当前验证分数是否优于之前的最佳验证结果（self.best_validation_result）。如果是，更新最佳验证结果和相应的检查点名称（self.best_cp_name）
                self.best_validation_result = reader_validation_score
                self.best_cp_name = cp_name
                logger.info("New Best validation checkpoint %s", cp_name)

    def validate(self):
        logger.info("Validation ...")
        cfg = self.cfg
        self.reader.eval()
        data_iterator = self.get_data_iterator(cfg.dev_files, cfg.train.dev_batch_size, False, shuffle=False)

        log_result_step = cfg.train.log_batch_step
        all_results = []

        eval_top_docs = cfg.eval_top_docs
        for i, samples_batch in enumerate(data_iterator.iterate_ds_data()):
            input = create_reader_input(
                self.tensorizer.get_pad_id(),
                samples_batch,
                cfg.passages_per_question_predict,
                cfg.encoder.sequence_length,
                cfg.max_n_answers,
                is_train=False,
                shuffle=False,
                sep_token_id=self.tensorizer.tokenizer.sep_token_id,  # TODO: tmp
            )

            input = ReaderBatch(**move_to_device(input._asdict(), cfg.device))
            attn_mask = self.tensorizer.get_attn_mask(input.input_ids)

            with torch.no_grad():
                start_logits, end_logits, relevance_logits = self.reader(
                    input.input_ids, attn_mask, input.token_type_ids
                )

            batch_predictions = self._get_best_prediction(
                start_logits,
                end_logits,
                relevance_logits,
                samples_batch,
                passage_thresholds=eval_top_docs,
            )

            all_results.extend(batch_predictions)

            if (i + 1) % log_result_step == 0:
                logger.info("Eval step: %d ", i)

        ems = defaultdict(list)

        for q_predictions in all_results:
            gold_answers = q_predictions.gold_answers
            span_predictions = q_predictions.predictions  # {top docs threshold -> SpanPrediction()}
            for (n, span_prediction) in span_predictions.items():
                em_hit = max([exact_match_score(span_prediction.prediction_text, ga) for ga in gold_answers])
                ems[n].append(em_hit)
        em = 0
        for n in sorted(ems.keys()):
            em = np.mean(ems[n])
            logger.info("n=%d\tEM %.2f" % (n, em * 100))

        if cfg.prediction_results_file:
            self._save_predictions(cfg.prediction_results_file, all_results)

        return em

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        eval_step: int,
        train_data_iterator: ShardedDataIterator,
        global_step: int,
    ):
        """ 
        scheduler: 学习率调度器，用于调整训练过程中的学习率。
        epoch: 当前的训练周期数。
        eval_step: 指定多少步进行一次评估。
        train_data_iterator: 分片数据迭代器，用于遍历训练数据。
        global_step: 全局步数，表示从训练开始至当前已经执行的更新次数。
        """
        # 初始化一些局部变量，如 epoch_loss（周期损失）和 rolling_train_loss（滚动损失）
        cfg = self.cfg
        rolling_train_loss = 0.0
        epoch_loss = 0
        log_result_step = cfg.train.log_batch_step
        rolling_loss_step = cfg.train.train_rolling_loss_step

        self.reader.train() # 准备模型进行训练
        epoch_batches = train_data_iterator.max_iterations

        # 遍历训练数据集
        for i, samples_batch in enumerate(train_data_iterator.iterate_ds_data(epoch=epoch)):

            data_iteration = train_data_iterator.get_iteration()

            # enables to resume to exactly same train state
            # 如果设置了完全可恢复训练的标志 cfg.fully_resumable，则根据当前的全局步数和种子值设置随机种子，以确保训练的可复现性
            if cfg.fully_resumable:
                np.random.seed(cfg.seed + global_step)
                torch.manual_seed(cfg.seed + global_step)
                if cfg.n_gpu > 0:
                    torch.cuda.manual_seed_all(cfg.seed + global_step)

            # 处理数据批次，准备模型输入
            input = create_reader_input(
                self.tensorizer.get_pad_id(),
                samples_batch,
                cfg.passages_per_question,
                cfg.encoder.sequence_length,
                cfg.max_n_answers,
                is_train=True,
                shuffle=True,
                sep_token_id=self.tensorizer.tokenizer.sep_token_id,  # TODO: tmp
            )

            loss = self._calc_loss(input) # 计算损失

            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            max_grad_norm = cfg.train.max_grad_norm
            # 损失反向传播：根据是否使用混合精度训练 (cfg.fp16)，选择使用 amp.scale_loss（用于混合精度）或直接调用 loss.backward()
            if cfg.fp16:
                from apex import amp

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), max_grad_norm)
            else:
                loss.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.reader.parameters(), max_grad_norm)

            # 每 cfg.train.gradient_accumulation_steps 步执行一次参数更新和学习率调度
            if (i + 1) % cfg.train.gradient_accumulation_steps == 0:
                self.optimizer.step()
                scheduler.step()
                self.reader.zero_grad()
                global_step += 1

            # 每 log_result_step 步记录训练日志，包括当前周期、步数、全局步数和学习率
            if i % log_result_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch: %d: Step: %d/%d, global_step=%d, lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    global_step,
                    lr,
                )

            # 每 rolling_loss_step 步计算并记录当前的平均滚动损失
            if (i + 1) % rolling_loss_step == 0:
                logger.info("Train batch %d", data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info(
                    "Avg. loss per last %d batches: %f",
                    rolling_loss_step,
                    latest_rolling_train_av_loss,
                )
                rolling_train_loss = 0.0

            # 每 eval_step 步进行一次评估，并保存模型
            if global_step % eval_step == 0:
                logger.info(
                    "Validation: Epoch: %d Step: %d/%d",
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(epoch, train_data_iterator.get_iteration(), scheduler)
                self.reader.train()

        # 在周期结束时，计算并记录整个周期的平均损失
        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        return global_step

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        """ 
        不仅保存了模型的状态，还保存了优化器和调度器的状态，以及当前的训练纪元和批次偏移量，这样训练就可以从这个检查点恢复继续执行
        
        scheduler: 当前使用的学习率调度器。
        epoch: 当前训练的纪元。
        offset: 当前纪元中的批次偏移量。
        """
        # 从实例变量 cfg 获取配置信息，并通过 get_model_obj 函数获取要保存的模型对象
        cfg = self.cfg
        model_to_save = get_model_obj(self.reader)
        # 构造检查点文件的路径。路径由输出目录 (cfg.output_dir)、检查点文件名前缀 (cfg.checkpoint_file_name)，以及纪元和偏移量的信息组成，以确保每个检查点文件名都是唯一的
        cp = os.path.join(
            cfg.output_dir,
            cfg.checkpoint_file_name + "." + str(epoch) + ("." + str(offset) if offset > 0 else ""),
        )

        # 从配置中提取模型的元数据参数。这些参数可能包括模型的特定配置，如预训练模型名称、序列长度等
        meta_params = get_encoder_params_state_from_cfg(cfg)

        # 创建一个 CheckpointState 实例，其中包含模型的状态、优化器的状态、调度器的状态、当前的偏移量、纪元和元数据参数。这个状态对象提供了从检查点恢复训练所需的所有信息
        state = CheckpointState(
            model_to_save.state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        # 使用 torch.save 函数将 CheckpointState 实例的字典表示形式保存到构造的文件路径下。这样就创建了一个检查点文件，它包含了模型训练状态的快照
        torch.save(state._asdict(), cp)
        return cp # 返回保存的检查点文件的路径

    def _load_saved_state(self, saved_state: CheckpointState):
        # 当训练因为某些原因（如电力故障、计算资源被回收等）被中断时，可以从之前保存的状态恢复训练，而不是从头开始
        epoch = saved_state.epoch
        offset = saved_state.offset # 当前 epoch 中的批次（batch）编号
        if offset == 0:  # epoch has been completed
            epoch += 1
        # 使用日志记录器（logger）输出当前加载的检查点的信息，包括批次（offset）和 epoch
        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)
        self.start_epoch = epoch # 记录开始的 epoch
        self.start_batch = offset # 记录开始的批次

        # 调用 get_model_obj 函数，传入 self.reader 作为参数，以获取模型对象，并将其赋值给 model_to_load
        model_to_load = get_model_obj(self.reader)
        if saved_state.model_dict:
            # 如果存在模型状态字典，使用日志记录器输出正在加载模型权重的信息
            logger.info("Loading model weights from saved state ...")
            # 调用 load_state_dict 方法将 saved_state.model_dict 中的权重加载到 model_to_load 模型中。参数 strict=False 允许部分匹配模型的参数
            model_to_load.load_state_dict(saved_state.model_dict, strict=False)

        # 使用日志记录器输出正在加载优化器状态的信息
        logger.info("Loading saved optimizer state ...")
        if saved_state.optimizer_dict:
            # 如果存在优化器状态字典，将其加载到当前的优化器中
            self.optimizer.load_state_dict(saved_state.optimizer_dict)
        # 将 saved_state 中的调度器状态字典 scheduler_dict 赋值给实例变量 scheduler_state，用于记录调度器的状态
        self.scheduler_state = saved_state.scheduler_dict

    def _get_best_prediction(
        self,
        start_logits,
        end_logits,
        relevance_logits,
        samples_batch: List[ReaderSample],
        passage_thresholds: List[int] = None,
    ) -> List[ReaderQuestionPredictions]:

        cfg = self.cfg
        max_answer_length = cfg.max_answer_length
        questions_num, passages_per_question = relevance_logits.size()

        _, idxs = torch.sort(
            relevance_logits,
            dim=1,
            descending=True,
        )

        batch_results = []
        for q in range(questions_num):
            sample = samples_batch[q]

            non_empty_passages_num = len(sample.passages)
            nbest = []
            for p in range(passages_per_question):
                passage_idx = idxs[q, p].item()
                if passage_idx >= non_empty_passages_num:  # empty passage selected, skip
                    continue
                reader_passage = sample.passages[passage_idx]
                sequence_ids = reader_passage.sequence_ids
                sequence_len = sequence_ids.size(0)
                # assuming question & title information is at the beginning of the sequence
                passage_offset = reader_passage.passage_offset

                p_start_logits = start_logits[q, passage_idx].tolist()[passage_offset:sequence_len]
                p_end_logits = end_logits[q, passage_idx].tolist()[passage_offset:sequence_len]

                ctx_ids = sequence_ids.tolist()[passage_offset:]
                best_spans = get_best_spans(
                    self.tensorizer,
                    p_start_logits,
                    p_end_logits,
                    ctx_ids,
                    max_answer_length,
                    passage_idx,
                    relevance_logits[q, passage_idx].item(),
                    top_spans=10,
                )
                nbest.extend(best_spans)
                if len(nbest) > 0 and not passage_thresholds:
                    break

            if passage_thresholds:
                passage_rank_matches = {}
                for n in passage_thresholds:
                    curr_nbest = [pred for pred in nbest if pred.passage_index < n]
                    passage_rank_matches[n] = curr_nbest[0]
                predictions = passage_rank_matches
            else:
                if len(nbest) == 0:
                    predictions = {passages_per_question: SpanPrediction("", -1, -1, -1, "")}
                else:
                    predictions = {passages_per_question: nbest[0]}
            batch_results.append(ReaderQuestionPredictions(sample.question, predictions, sample.answers))
        return batch_results

    def _calc_loss(self, input: ReaderBatch) -> torch.Tensor:
        # 接受一个类型为 ReaderBatch 的输入，这是一个封装了一批训练数据的对象，然后计算这批数据上模型的损失
        cfg = self.cfg # 包含了模型训练和评估过程中的各种配置信息
        # 将 input 对象的数据移动到指定的设备（如CPU或GPU）上。这里使用了一个函数 move_to_device，它接收一个字典形式的输入和一个目标设备，然后将数据移动到该设备上
        input = ReaderBatch(**move_to_device(input._asdict(), cfg.device))
        # 通过 tensorizer 对象获取注意力掩码（attention mask）。这个掩码用于指示哪些位置是有效的，哪些位置应该被模型忽略。input.input_ids 包含了输入数据的标识符
        attn_mask = self.tensorizer.get_attn_mask(input.input_ids)
        # 获取 input.input_ids 的维度信息，分别代表问题的数量、每个问题对应的段落数量和每个输入的长度
        questions_num, passages_per_question, _ = input.input_ids.size()

        if self.reader.training:
            # 在训练模式下，loss 是通过调用 reader 对象并传入输入数据和相关信息（如输入标识符、注意力掩码、令牌类型标识符、起始位置、结束位置和答案掩码）来计算得到的
            loss = self.reader(
                input.input_ids,
                attn_mask,
                input.token_type_ids,
                input.start_positions,
                input.end_positions,
                input.answers_mask,
            )

        else:
            # TODO: remove?
            # 如果不是训练模式，则在不计算梯度的上下文中计算 start_logits, end_logits, 和 rank_logits，并调用 compute_loss 函数计算损失
            with torch.no_grad():
                start_logits, end_logits, rank_logits = self.reader(input.input_ids, attn_mask, input.token_type_ids)

            loss = compute_loss(
                input.start_positions,
                input.end_positions,
                input.answers_mask,
                start_logits,
                end_logits,
                rank_logits,
                questions_num,
                passages_per_question,
            )
        if cfg.n_gpu > 1:
            # 如果使用的 GPU 数量大于1，则将损失值平均，以便在多GPU设置中进行一致的损失计算
            loss = loss.mean()
        if cfg.train.gradient_accumulation_steps > 1:
            # 如果梯度累积步数大于1，则相应地调整损失值。梯度累积允许在内存限制下使用更大的有效批量大小
            loss = loss / cfg.train.gradient_accumulation_steps

        return loss

    def _save_predictions(self, out_file: str, prediction_results: List[ReaderQuestionPredictions]):
        logger.info("Saving prediction results to  %s", out_file)
        with open(out_file, "w", encoding="utf-8") as output:
            save_results = []
            for r in prediction_results:
                save_results.append(
                    {
                        "question": r.id,
                        "gold_answers": r.gold_answers,
                        "predictions": [
                            {
                                "top_k": top_k,
                                "prediction": {
                                    "text": span_pred.prediction_text,
                                    "score": span_pred.span_score,
                                    "relevance_score": span_pred.relevance_score,
                                    "passage_idx": span_pred.passage_index,
                                    "passage": self.tensorizer.to_string(span_pred.passage_token_ids),
                                },
                            }
                            for top_k, span_pred in r.predictions.items()
                        ],
                    }
                )
            output.write(json.dumps(save_results, indent=4) + "\n")


@hydra.main(config_path="conf", config_name="extractive_reader_train_cfg")
def main(cfg: DictConfig):

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg)

    if cfg.local_rank in [-1, 0]:
        logger.info("CFG (after gpu  configuration):")
        logger.info("%s", OmegaConf.to_yaml(cfg))

    trainer = ReaderTrainer(cfg)

    if cfg.train_files is not None:
        trainer.run_train()
    elif cfg.dev_files:
        logger.info("No train files are specified. Run validation.")
        trainer.validate()
    else:
        logger.warning("Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do.")


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args
    main()
