#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Command line arguments utils
"""


import logging
import os
import random
import socket
import subprocess
from typing import Tuple

import numpy as np
import torch
from omegaconf import DictConfig

logger = logging.getLogger()

# TODO: to be merged with conf_utils.py


def set_cfg_params_from_state(state: dict, cfg: DictConfig):
    """
    从一个给定的状态对象state中覆盖一些编码器配置参数到配置对象cfg中
    接受两个参数：state是一个字典，包含了需要被覆盖到cfg的参数；cfg是一个DictConfig对象，通常由Hydra框架提供，用于管理配置信息
    """
    if not state:
        # 如果state为空，则函数直接返回，不执行任何操作
        return

    cfg.do_lower_case = state["do_lower_case"] # 从state中直接设置do_lower_case参数到cfg。这个参数通常用于指定在文本处理时是否将文本转换为小写

    if "encoder" in state:
        # 检查state中是否有encoder键，如果有，则表示需要更新编码器相关的配置
        saved_encoder_params = state["encoder"] # 获取state中的encoder部分，这里包含了编码器的参数
        # 在尝试直接将encoder的参数赋值给cfg.encoder时遇到了问题（cfg.encoder = state["encoder"]不起作用），原因可能与DictConfig的特性或限制有关
        # TODO: try to understand why cfg.encoder = state["encoder"] doesn't work

        for k, v in saved_encoder_params.items():
            # 通过遍历saved_encoder_params中的每个键值对，并使用setattr函数逐一将它们设置到cfg.encoder上，从而绕过了直接赋值的问题。这种方法允许动态地更新配置对象中嵌套的属性
            # TODO: tmp fix
            if k == "q_wav2vec_model_cfg":
                k = "q_encoder_model_cfg"
            if k == "q_wav2vec_cp_file":
                k = "q_encoder_cp_file"
            if k == "q_wav2vec_cp_file":
                k = "q_encoder_cp_file"

            setattr(cfg.encoder, k, v) # 设置对象的属性。如果属性已经存在于对象中，setattr会更新该属性的值；如果属性不存在，它将创建一个新的属性并赋值
    else:  # 'old' checkpoints backward compatibility support
        pass
        # cfg.encoder.pretrained_model_cfg = state["pretrained_model_cfg"]
        # cfg.encoder.encoder_model_type = state["encoder_model_type"]
        # cfg.encoder.pretrained_file = state["pretrained_file"]
        # cfg.encoder.projection_dim = state["projection_dim"]
        # cfg.encoder.sequence_length = state["sequence_length"]


def get_encoder_params_state_from_cfg(cfg: DictConfig):
    """
    从配置中选择需要在检查点中保存的参数值，以便训练过的模型可以用于下游任务而无需再次指定这些参数
    cfg: DictConfig：这是一个函数参数，表示配置字典。它使用了DictConfig类型，这通常与OmegaConf库一起使用，OmegaConf是用于处理配置文件的Python库
    """
    return {
        "do_lower_case": cfg.do_lower_case, # 指示文本处理时是否将所有字符转换为小写
        "encoder": cfg.encoder, # 指定哪种编码器被用于模型，比如BERT或GPT
    } 


def set_seed(args):
    # 设置随机种子，以确保实验的可重复性
    # args：这是一个函数参数，通常包含了运行实验所需的各种设置，其中就包括了随机种子seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 为PyTorch库的随机数生成器设置种子
    if args.n_gpu > 0: # 检查是否有GPU可用（n_gpu大于0表示有GPU）
        torch.cuda.manual_seed_all(seed) # 如果有GPU可用，这一行代码为所有CUDA设备设置相同的随机种子，以确保在GPU上也有可重复的实验结果


def setup_cfg_gpu(cfg):
    """
    定义了一个名为setup_cfg_gpu的函数，该函数接受一个参数cfg，这个参数包含了训练的配置信息
    Setup params for CUDA, GPU & distributed training
    """
    logger.info("CFG's local_rank=%s", cfg.local_rank) # 记录配置中的local_rank值，这通常用于分布式训练，表示当前进程的本地排名
    ws = os.environ.get("WORLD_SIZE") # 从环境变量中获取WORLD_SIZE的值，它代表参与训练的总进程数
    cfg.distributed_world_size = int(ws) if ws else 1 # 设置distributed_world_size为环境变量WORLD_SIZE的值，如果未设置，则默认为1
    logger.info("Env WORLD_SIZE=%s", ws) # 记录环境变量WORLD_SIZE的值

    if cfg.distributed_port and cfg.distributed_port > 0:
        # 如果指定了distributed_port且其值大于0，代码会尝试从SLURM参数初始化分布式训练模式。SLURM是一种常用于高性能计算集群的作业调度系统
        logger.info("distributed_port is specified, trying to init distributed mode from SLURM params ...")
        # 通过调用_infer_slurm_init(cfg)函数（这个函数在代码片段中未给出）来推断初始化方法、本地排名、世界大小以及设备ID
        init_method, local_rank, world_size, device = _infer_slurm_init(cfg)

        logger.info(
            "Inferred params from SLURM: init_method=%s | local_rank=%s | world_size=%s",
            init_method,
            local_rank,
            world_size,
        )

        cfg.local_rank = local_rank
        cfg.distributed_world_size = world_size
        cfg.n_gpu = 1

        torch.cuda.set_device(device) # 设置当前进程的GPU设备
        device = str(torch.device("cuda", device))

        # 初始化进程组，这是分布式训练的关键步骤，它让不同的进程能够互相通信
        torch.distributed.init_process_group(
            backend="nccl", init_method=init_method, world_size=world_size, rank=local_rank
        )

    elif cfg.local_rank == -1 or cfg.no_cuda:  # 如果cfg.local_rank为-1或者设置了cfg.no_cuda，表示使用单节点多GPU模式或者CPU模式
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))  # 根据是否可用和配置决定使用CUDA设备还是CPU设备
        cfg.n_gpu = torch.cuda.device_count() # 获取可用的GPU数量
    else:  # 如果不是上述两种情况，则认为是在分布式模式下运行，但不是通过SLURM初始化的
        # 设置当前进程的CUDA设备，并初始化进程组，这同样是为了让不同的进程能够进行通信
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        torch.distributed.init_process_group(backend="nccl")
        cfg.n_gpu = 1

    cfg.device = device # 不论哪种模式，最后都会设置cfg.device为当前使用的设备

    # 记录最终的设备和分布式训练的相关信息，比如主机名、本地排名、使用的设备、GPU数量以及世界大小等
    logger.info(
        "Initialized host %s as d.rank %d on device=%s, n_gpu=%d, world size=%d",
        socket.gethostname(),
        cfg.local_rank,
        cfg.device,
        cfg.n_gpu,
        cfg.distributed_world_size,
    )
    
    logger.info("16-bits training: %s ", cfg.fp16) # 记录是否启用了16位训练(cfg.fp16)
    return cfg # 函数最后返回更新后的配置对象cfg


def _infer_slurm_init(cfg) -> Tuple[str, int, int, int]:
    # 为了在使用SLURM作业调度器环境中，根据环境变量和SLURM的设置来推断分布式训练的初始化方法、本地排名（local_rank）、世界大小（world_size）、以及设备ID（device_id）
    # 函数接受一个配置对象cfg作为输入，返回一个包含四个元素的元组，分别对应于初始化方法（字符串），本地排名（整数），世界大小（整数），和设备ID（整数）

    node_list = os.environ.get("SLURM_STEP_NODELIST")
    # 函数尝试从环境变量中获取SLURM相关的节点列表。它先检查SLURM_STEP_NODELIST，如果未找到，则检查SLURM_JOB_NODELIST
    if node_list is None:
        node_list = os.environ.get("SLURM_JOB_NODELIST")
    logger.info("SLURM_JOB_NODELIST: %s", node_list)

    if node_list is None:
        raise RuntimeError("Can't find SLURM node_list from env parameters")

    local_rank = None
    world_size = None
    distributed_init_method = None
    device_id = None
    try:
        # 如果成功获取到节点列表，函数将使用scontrol命令来查询这些节点的主机名，并构造出一个初始化方法的字符串，这个字符串以tcp://开头，后面跟着主机名和分配给分布式训练任务的端口号
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", node_list])
        distributed_init_method = "tcp://{host}:{port}".format(
            host=hostnames.split()[0].decode("utf-8"),
            port=cfg.distributed_port,
        )
        nnodes = int(os.environ.get("SLURM_NNODES"))
        logger.info("SLURM_NNODES: %s", nnodes)
        ntasks_per_node = os.environ.get("SLURM_NTASKS_PER_NODE")
        # 函数还会计算world_size（参与训练的总进程数）和local_rank（当前进程在所有进程中的排名）。这些计算依赖于多个SLURM环境变量，如SLURM_NNODES（节点数），SLURM_NTASKS_PER_NODE（每节点任务数），SLURM_NTASKS（总任务数），SLURM_NODEID（当前节点ID），SLURM_PROCID（进程ID），和SLURM_LOCALID（在当前节点上的本地进程ID）
        if ntasks_per_node is not None:
            ntasks_per_node = int(ntasks_per_node)
            logger.info("SLURM_NTASKS_PER_NODE: %s", ntasks_per_node)
        else:
            ntasks = int(os.environ.get("SLURM_NTASKS"))
            logger.info("SLURM_NTASKS: %s", ntasks)
            assert ntasks % nnodes == 0
            ntasks_per_node = int(ntasks / nnodes)

        if ntasks_per_node == 1:
            gpus_per_node = torch.cuda.device_count()
            node_id = int(os.environ.get("SLURM_NODEID"))
            local_rank = node_id * gpus_per_node
            world_size = nnodes * gpus_per_node
            logger.info("node_id: %s", node_id)
        else:
            world_size = ntasks_per_node * nnodes
            proc_id = os.environ.get("SLURM_PROCID")
            local_id = os.environ.get("SLURM_LOCALID")
            logger.info("SLURM_PROCID %s", proc_id)
            logger.info("SLURM_LOCALID %s", local_id)
            local_rank = int(proc_id)
            device_id = int(local_id)

    except subprocess.CalledProcessError as e:  # scontrol failed
        # 如果在调用scontrol过程中遇到CalledProcessError异常，说明scontrol命令执行失败，函数会直接将这个异常抛出
        raise e
    except FileNotFoundError:  # Slurm is not installed
        # 如果是因为SLURM未安装导致的FileNotFoundError异常，则函数不做任何操作（pass）
        pass
    # 函数返回由distributed_init_method（分布式初始化方法），local_rank（本地排名），world_size（世界大小），和device_id（设备ID）组成的元组
    return distributed_init_method, local_rank, world_size, device_id


def setup_logger(logger):
    # 配置日志记录器(logger)以便于输出日志信息
    # 定义了一个名为setup_logger的函数，它接收一个logger对象作为参数。这个logger对象是需要被配置的日志记录器
    logger.setLevel(logging.INFO) # 设置logger的日志级别为INFO。这意味着所有INFO级别及以上（如WARNING, ERROR, CRITICAL）的日志信息都将被记录，而更低级别的日志（如DEBUG）将被忽略
    if logger.hasHandlers(): # 检查logger是否已经有配置的日志处理器（handlers）
        # 如果有，那么清除所有已经添加的处理器。这是为了避免重复记录日志信息，特别是在多次调用setup_logger函数时
        logger.handlers.clear()
    log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s") # 定义一个日志格式器(log_formatter)，并指定日志信息的格式。在这个格式中，每条日志信息将包括线程ID、时间戳、日志级别、记录器名称和日志消息本身
    console = logging.StreamHandler() # 创建一个日志处理器console，这个处理器会将日志信息输出到标准输出流（即通常的控制台）
    console.setFormatter(log_formatter) # 将之前定义的日志格式器应用到console处理器上，这样console处理器输出的所有日志信息都将按照指定的格式进行格式化
    logger.addHandler(console) # 将console处理器添加到logger上，这样配置完成后，logger就能够通过console处理器按照指定的格式输出日志信息到控制台了
