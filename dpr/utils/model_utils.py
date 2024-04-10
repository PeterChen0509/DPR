#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import glob
import logging
import os
from typing import List

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.serialization import default_restore_location

logger = logging.getLogger()

CheckpointState = collections.namedtuple(
    "CheckpointState",
    [
        "model_dict",   # 保存模型的参数
        "optimizer_dict", # 优化器的状态字典
        "scheduler_dict", # 学习率调度器的状态字典
        "offset", # 记录当前处理到哪个数据点，以便恢复训练时从正确的位置开始
        "epoch",    # 代表当前的训练轮数
        "encoder_params",   # 保存这些参数可以确保编码器的状态可以被准确地恢复
    ],
)


def setup_for_distributed_mode(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: object,
    n_gpu: int = 1,
    local_rank: int = -1,
    fp16: bool = False,
    fp16_opt_level: str = "O1",
) -> (nn.Module, torch.optim.Optimizer):
    """ 
    配置分布式模式的函数，它用于准备一个深度学习模型以便在多个GPU或多个节点上进行训练
    该函数接受多个参数，包括模型(model)、优化器(optimizer)、设备(device)、GPU数量(n_gpu)、本地排名(local_rank)、是否使用半精度浮点数(fp16)以及半精度优化级别(fp16_opt_level)
    """
    model.to(device)    # 将模型移动到指定的设备上，这里的设备可以是CPU或GPU
    if fp16:    # 检查是否启用了半精度(fp16)训练。如果是，执行下面的代码块
        try:
            #  尝试导入apex库。apex是NVIDIA开发的一个库，它提供了混合精度训练和分布式训练的工具
            import apex
            from apex import amp

            apex.amp.register_half_function(torch, "einsum")    # 为PyTorch的einsum函数注册半精度功能，以确保它在半精度训练中正确工作
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        # 使用amp初始化模型和优化器，设置半精度优化级别
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if n_gpu > 1:
        # 如果使用的GPU数量大于1，则执行下面的代码块
        model = torch.nn.DataParallel(model)    # 使用DataParallel对模型进行封装，以便在多个GPU上并行处理数据和训练模型

    if local_rank != -1:
        #  如果指定了本地排名（通常在分布式训练环境中使用），则执行下面的代码块
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device if device else local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )   # 指定设备ID和输出设备，并设置find_unused_parameters=True来找到未使用的参数，这对于一些特定的模型结构非常有用
    return model, optimizer


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)


def move_to_device(sample, device):
    # 将数据样本从当前设备移动到指定的设备上
    # 接受两个参数：sample（一个数据样本，可以是多种数据结构的任意一种，例如张量、字典、列表或元组）和 device（目标设备，通常是一个字符串，如 'cpu' 或 'cuda:0'）
    if len(sample) == 0:
        # 检查 sample 的长度是否为 0。如果是，函数直接返回一个空字典 {}。这是为了处理空样本的边缘情况，确保函数在接收到空输入时不会出错
        return {}

    def _move_to_device(maybe_tensor, device):
        # 递归地处理 sample 中的每个元素，并将其移动到指定的设备上。这个函数同样接受两个参数：maybe_tensor（可能是一个张量或其他数据结构）和 device
        if torch.is_tensor(maybe_tensor):
            # 判断 maybe_tensor 是否为一个 PyTorch 张量。如果是，使用 .to(device) 方法将张量移动到指定设备，并返回结果
            return maybe_tensor.to(device)
        elif isinstance(maybe_tensor, dict):
            # 如果 maybe_tensor 是一个字典，那么递归地对字典的每个值应用 _move_to_device 函数，并构建一个新的字典，其键不变，值为移动到指定设备上的结果
            return {key: _move_to_device(value, device) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            #  如果 maybe_tensor 是一个列表，那么递归地对列表中的每个元素应用 _move_to_device 函数，并返回一个新列表，包含移动到指定设备上的元素
            return [_move_to_device(x, device) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            #  如果 maybe_tensor 是一个元组，处理方式与列表相似，但由于元组是不可变的，所以返回的是一个列表而不是元组。这可能是一个需要注意的地方，因为这改变了原始数据结构的类型
            return [_move_to_device(x, device) for x in maybe_tensor]
        else:
            #  如果 maybe_tensor 不是以上任何一种类型，那么它可能是一个数值或其他非张量类型的数据，直接返回它本身，因为这些类型的数据不需要移动到特定的设备上
            return maybe_tensor

    #  在函数的最后，调用嵌套定义的 _move_to_device 函数，并将其返回值作为 move_to_device 函数的返回值。这样，无论 sample 的原始结构如何，这个函数都能正确地处理并将其内部的所有张量移动到指定的设备上
    return _move_to_device(sample, device)


def get_schedule_linear(
    optimizer,
    warmup_steps,
    total_training_steps,
    steps_shift=0,
    last_epoch=-1,
):

    """
    创建一个学习率调度策略，该策略在预热期间学习率线性增加，之后线性减少，直到训练结束。这种调度方式对于许多训练深度学习模型非常有用，因为它有助于模型在训练初期快速收敛，同时避免训练后期的过拟合
    定义函数，接受优化器（optimizer）、预热步数（warmup_steps）、总训练步数（total_training_steps）、步数偏移（steps_shift，默认为0）和最后一个epoch的索引（last_epoch，默认为-1）作为参数
    """

    def lr_lambda(current_step):
        # 定义一个内部函数lr_lambda，它根据当前步骤（current_step）计算学习率的比例因子
        current_step += steps_shift # 将当前步骤调整以考虑步数偏移
        if current_step < warmup_steps:
            # 如果当前步骤在预热期间，则计算学习率的比例因子为当前步骤与预热步数的比例
            # 返回学习率的比例因子，确保分母至少为1，避免除以0的情况
            return float(current_step) / float(max(1, warmup_steps))
        # 这意味着在预热期之后，学习率将根据剩余训练步数线性减少，直到训练结束
        # 确保返回的比例因子不会低于一个很小的正数（1e-7），避免学习率完全降为零
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )

    # 使用内部定义的lr_lambda函数作为学习率调度函数，创建并返回一个LambdaLR调度器实例。LambdaLR是PyTorch中的一个工具，允许用户定义自己的学习率调度策略。optimizer是被调整学习率的优化器，lr_lambda是一个接受当前epoch索引并返回学习率乘法因子的函数，last_epoch指的是最后一个epoch的索引，用于重新开始训练时的调度恢复
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def init_weights(modules: List):
    for module in modules:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def get_model_obj(model: nn.Module):
    # 获取PyTorch模型的实际对象。在使用诸如DataParallel或DistributedDataParallel之类的PyTorch模块进行多GPU训练时，原始模型会被封装在一个叫做module的属性里。这个函数检查传入的模型对象model是否有module这个属性（即是否被封装过）
    return model.module if hasattr(model, "module") else model


def get_model_file(args, file_prefix) -> str:
    # 获取模型文件的路径。它接收两个参数：一个包含各种命令行参数的对象args和一个文件前缀file_prefix
    if args.model_file and os.path.exists(args.model_file):
        # 首先检查args.model_file是否存在且指向一个实际的文件。如果是，直接返回该文件的路径
        return args.model_file

    # 如果上述条件不满足，函数会构造一个搜索路径，搜索args.output_dir目录下所有以file_prefix开头的文件。这通过使用glob.glob函数完成
    out_cp_files = glob.glob(os.path.join(args.output_dir, file_prefix + "*")) if args.output_dir else []
    logger.info("Checkpoint files %s", out_cp_files) # 使用logger.info记录找到的所有符合条件的文件路径
    model_file = None

    # 如果找到了一个或多个符合条件的文件，函数通过比较文件的创建时间（os.path.getctime），选取最新的一个文件作为模型文件，并将其路径赋值给model_file
    if len(out_cp_files) > 0:
        model_file = max(out_cp_files, key=os.path.getctime)
    return model_file


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    # 从一个保存有模型状态的文件中加载状态，并将这些状态封装在一个 CheckpointState 命名元组中返回
    logger.info("Reading saved model from %s", model_file)
    # 使用 torch.load 函数加载模型文件。torch.load 的 map_location 参数指定了加载模型时张量的设备位置。这里，通过一个匿名函数 lambda s, l: default_restore_location(s, "cpu")，指示将所有张量映射到 CPU。这是为了确保无论模型最初是在哪种设备（比如 GPU）上保存的，都能在没有该设备的情况下被加载。default_restore_location 是一个假定存在的函数，用于指定恢复位置的默认设备。
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, "cpu"))
    logger.info("model_state_dict keys %s", state_dict.keys()) # 验证加载了哪些状态和进行调试
    # 使用加载的状态字典 state_dict 中的键值对作为参数，创建一个 CheckpointState 实例，并返回这个实例。这里使用的 **state_dict 语法是 Python 的关键字参数解包，它允许你将一个字典的键值对直接传递给函数的参数
    return CheckpointState(**state_dict)
