#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import logging
import math
import os
import pathlib
import pickle
from typing import List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from dpr.data.biencoder_data import BiEncoderPassage
from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu, setup_logger

from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)

logger = logging.getLogger()
setup_logger(logger)


def gen_ctx_vectors(
    cfg: DictConfig,
    ctx_rows: List[Tuple[object, BiEncoderPassage]],
    model: nn.Module,
    tensorizer: Tensorizer,
    insert_title: bool = True,
) -> List[Tuple[object, np.array]]:
    # 生成上下文向量（context vectors）的函数
    """ 
    cfg: DictConfig是一个配置对象，包含了执行此函数所需的各种配置，如批处理大小、设备信息等。
    ctx_rows: List[Tuple[object, BiEncoderPassage]]是一个列表，其中的每个元素都是一个元组，包含了上下文的ID（任意对象）和一个BiEncoderPassage对象（包含文本和标题等信息）。
    model: nn.Module是用于生成向量表示的神经网络模型。
    tensorizer: Tensorizer是一个用于文本处理的工具，能将文本转换成模型可处理的张量格式。
    insert_title: bool = True是一个布尔值，指定是否在文本中插入标题。
    """
    n = len(ctx_rows)
    bsz = cfg.strict_batch_size # 从配置中获取批处理大小
    total = 0 # 记录已处理的上下文数量
    results = [] # 最终要返回的结果列表
    for j, batch_start in enumerate(range(0, n, bsz)):
        # 使用enumerate和range函数分批处理上下文，batch_start是每个批次的起始索引
        batch = ctx_rows[batch_start : batch_start + bsz]
        # 为当前批次的每个上下文生成张量表示，考虑是否加入标题。
        batch_token_tensors = [
            tensorizer.text_to_tensor(ctx[1].text, title=ctx[1].title if insert_title else None) for ctx in batch
        ]

        # 将这些张量合并并移动到指定的设备上（比如GPU）
        ctx_ids_batch = move_to_device(torch.stack(batch_token_tensors, dim=0), cfg.device)
        ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), cfg.device)
        # 注意力掩码用于模型中，标记哪些部分是有效输入
        ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), cfg.device)
        # 在torch.no_grad()上下文中调用模型，避免计算梯度，以提高效率
        with torch.no_grad():
            _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
        out = out.cpu() # 将模型输出转换回CPU，准备后续处理

        ctx_ids = [r[0] for r in batch]
        extra_info = []
        # 如果批次中的元素包含额外信息（即元组长度大于3），则将这些信息包含在结果中
        if len(batch[0]) > 3:
            extra_info = [r[3:] for r in batch]

        assert len(ctx_ids) == out.size(0)
        total += len(ctx_ids)

        # 根据是否有额外信息，构建最终的结果列表，每个元素包括上下文ID、向量表示，以及可能的额外信息
        if extra_info:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy(), *extra_info[i]) for i in range(out.size(0))])
        else:
            results.extend([(ctx_ids[i], out[i].view(-1).numpy()) for i in range(out.size(0))])

        # 每处理10个上下文，记录一次日志
        if total % 10 == 0:
            logger.info("Encoded passages %d", total)
    # 函数返回一个列表，包含了所有上下文的ID和它们的向量表示（以及可能的额外信息）
    return results


# 使用@hydra.main装饰器定义main函数，指定Hydra配置的路径和名称。这使得函数可以自动读取配置文件，配置文件中包含了运行这个脚本所需要的所有参数
@hydra.main(config_path="conf", config_name="gen_embs")
def main(cfg: DictConfig):

    # 确保必要的配置参数model_file（模型文件）和ctx_src（文本片段的来源）已经被指定。如果这些参数未被提供，脚本将断言错误并提示用户
    assert cfg.model_file, "Please specify encoder checkpoint as model_file param"
    assert cfg.ctx_src, "Please specify passages source as ctx_src param"

    # 通过setup_cfg_gpu(cfg)函数调整配置，以适应当前的GPU设置，该函数可能涉及到选择正确的设备、分配GPU等操作
    cfg = setup_cfg_gpu(cfg)

    # 从指定的模型文件中加载状态，这包括模型权重和参数
    saved_state = load_states_from_checkpoint(cfg.model_file)
    # 使用保存的状态中的编码器参数来更新配置
    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    # 打印当前配置信息到日志，OmegaConf.to_yaml(cfg)将配置对象转换为YAML格式的字符串，方便查看和记录
    logger.info("CFG:")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    # 初始化双向编码器组件。根据配置中指定的编码器类型，这可能包括文本的向量化处理、编码器模型本身等
    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    # 根据配置（cfg.encoder_type）决定使用上下文（ctx）编码器模型还是问题（question）编码器模型
    encoder = encoder.ctx_model if cfg.encoder_type == "ctx" else encoder.question_model

    # 将编码器模型配置为分布式运行模式（如果适用），包括设置设备、多GPU支持、分布式训练的本地排名、半精度浮点数（fp16）配置等
    encoder, _ = setup_for_distributed_mode(
        encoder,
        None,
        cfg.device,
        cfg.n_gpu,
        cfg.local_rank,
        cfg.fp16,
        cfg.fp16_opt_level,
    )
    encoder.eval()  # 将模型设置为评估模式，这通常意味着在模型中禁用了诸如dropout之类的随机性

    # 从保存的模型状态中加载权重到编码器模型。
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")
    logger.debug("saved model keys =%s", saved_state.model_dict.keys())

    # 这里特别处理了上下文模型的权重，只加载以"ctx_model."为前缀的键值对
    prefix_len = len("ctx_model.")
    ctx_state = {
        key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if key.startswith("ctx_model.")
    }
    model_to_load.load_state_dict(ctx_state, strict=False)

    logger.info("reading data source: %s", cfg.ctx_src) # 打印日志信息，指出正在读取数据来源

    # 通过Hydra实用工具实例化数据源对象，cfg.ctx_sources[cfg.ctx_src]指的是配置文件中定义的数据源
    ctx_src = hydra.utils.instantiate(cfg.ctx_sources[cfg.ctx_src])
    # 加载数据到字典all_passages_dict中，然后将字典转换为元组列表all_passages，每个元组包含一个键值对，键是文本片段的ID，值是文本片段本身
    all_passages_dict = {}
    ctx_src.load_data_to(all_passages_dict)
    all_passages = [(k, v) for k, v in all_passages_dict.items()]

    # 计算每个数据分片的大小，并根据配置中的shard_id确定当前处理的分片的起始和结束索引
    shard_size = math.ceil(len(all_passages) / cfg.num_shards)
    start_idx = cfg.shard_id * shard_size
    end_idx = start_idx + shard_size

    # 打印日志信息，指出当前处理的文本片段的范围
    logger.info(
        "Producing encodings for passages range: %d to %d (out of total %d)",
        start_idx,
        end_idx,
        len(all_passages),
    )
    shard_passages = all_passages[start_idx:end_idx]    # 从总的文本片段列表中提取当前分片的片段

    # 使用gen_ctx_vectors函数为当前分片生成嵌入向量，这个函数可能涉及到将文本通过编码器模型转换为嵌入向量的过程
    data = gen_ctx_vectors(cfg, shard_passages, encoder, tensorizer, True)

    # 根据配置中的out_file参数和shard_id来构建输出文件的名称，并确保输出文件的目录存在
    file = cfg.out_file + "_" + str(cfg.shard_id)
    # 将生成的嵌入向量数据序列化并保存到指定的文件中, 确保目录存在，如果不存在则创建
    pathlib.Path(os.path.dirname(file)).mkdir(parents=True, exist_ok=True)
    # 打印日志信息，汇报处理的文本片段数量和结果文件的位置
    logger.info("Writing results to %s" % file)
    with open(file, mode="wb") as f:
        pickle.dump(data, f)

    logger.info("Total passages processed %d. Written to %s", len(data), file)

# 将一系列文本片段通过预训练的编码器模型转换为嵌入向量，并将结果保存到文件中

if __name__ == "__main__":
    main()
