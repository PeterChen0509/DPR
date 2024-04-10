#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for distributed model training
"""

import pickle

import torch
import torch.distributed as dist
# 简化在使用PyTorch进行分布式训练时的常见操作。分布式训练允许多个处理器或计算节点并行处理数据，加速训练过程。这些函数依赖于PyTorch的torch.distributed模块，它提供了用于跨多个进程进行通信的基本原语


def get_rank():
    # 返回当前进程的排名（rank）。在分布式训练中，每个进程都被分配了一个唯一的排名（从0开始），用于标识不同的进程
    return dist.get_rank()


def get_world_size():
    # 返回在当前分布式训练任务中参与计算的总进程数（即“world size”）
    return dist.get_world_size()


def get_default_group():
    # 返回默认的进程组，即dist.group.WORLD。在PyTorch分布式训练中，进程组是一种逻辑分组方式，允许在组内的所有进程之间进行通信
    return dist.group.WORLD


def all_reduce(tensor, group=None):
    # 执行一个全局的归约操作，将来自不同进程的tensor值累加（或其他操作）后，结果被分发到所有进程。这是并行计算中常见的一种操作，用于合并来自不同进程的结果
    """ 
    参数tensor是要进行归约操作的张量。
    参数group指定了要在哪个进程组内执行归约操作。如果未指定，则使用默认进程组（即所有进程）。
    归约操作默认使用加法，但torch.distributeded支持多种归约操作，如乘法、最大值和最小值等
    """
    if group is None:
        group = get_default_group()
    return dist.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """
    函数的目的是从所有节点收集数据到一个列表中，其中每个节点的数据可以是任意的Python对象（必须是可序列化的）
    data: 本地工作节点要收集的数据。这个数据会被发送到所有工作节点上。
    group (可选): 用于集体通信的进程组。
    max_size: 每个节点发送数据的最大字节大小，默认为16384字节。
    """
    SIZE_STORAGE_BYTES = 4  # int32 to encode the payload size

    enc = pickle.dumps(data) # 使用pickle.dumps将数据序列化为字节串, 这个函数主要用于将Python对象转换为可以被存储或通过网络传输的格式。使用pickle.dumps时，你不需要打开一个文件来存储序列化后的数据，而是直接在内存中完成这一操作
    enc_size = len(enc)

    if enc_size + SIZE_STORAGE_BYTES > max_size:
        # 检查序列化后的数据大小是否超出了max_size。如果超出，将抛出ValueError
        raise ValueError(
            'encoded data exceeds max_size, this can be fixed by increasing buffer size: {}'.format(enc_size))

    rank = get_rank()
    world_size = get_world_size()
    buffer_size = max_size * world_size

    # 根据所有节点的最大可能需要的缓冲区大小（max_size * world_size），创建或重用一个足够大的ByteTensor缓冲区来存放所有节点的数据。此外，创建一个CPU缓冲区用于临时存储本地数据
    """
    _buffer和_cpu_buffer是附加在all_gather_list函数上的属性，用于存储临时数据。Python允许动态地给函数和对象添加属性，这里_buffer用于存储GPU上的数据，而_cpu_buffer用于存储CPU上的数据。这些缓冲区在执行分布式收集操作时被用来临时存储序列化后的数据
     hasattr(object, name)是Python的内置函数，用于判断对象是否有名为name的属性。
    .numel()是PyTorch张量（Tensor）的一个方法，返回张量中元素的总数。这里使用all_gather_list._buffer.numel()来检查_buffer张量中是否有足够的空间来存储当前需要的数据量
    torch.cuda.ByteTensor(size)创建一个指定大小的ByteTensor（字节张量），并将其存储在GPU上。这种类型的张量用于存储字节数据，每个元素占用1字节空间。
    torch.ByteTensor(size)创建一个指定大小的ByteTensor，但存储在CPU上。和GPU上的ByteTensor类似，它也是用于存储字节数据
    .pin_memory()是PyTorch张量的一个方法，它用于“固定”CPU上的张量内存
    """
    # 这段代码主要是为all_gather_list函数准备了两个缓冲区：_buffer用于在GPU上存储数据，而_cpu_buffer则存储在CPU上，并通过调用.pin_memory()使其成为固定内存，以优化CPU到GPU的数据传输性能。通过这种方式，all_gather_list函数可以更高效地处理分布式环境下的数据收集操作。
    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()

    buffer = all_gather_list._buffer
    buffer.zero_() #  清零 GPU 上的缓冲区，为新数据做准备
    cpu_buffer = all_gather_list._cpu_buffer

    # 断言编码后对象的大小（enc_size）小于允许的最大大小，这是通过 SIZE_STORAGE_BYTES 计算得出的
    assert enc_size < 256 ** SIZE_STORAGE_BYTES, 'Encoded object size should be less than {} bytes'.format(
        256 ** SIZE_STORAGE_BYTES)

    # 将对象大小（enc_size）编码为字节，并存储到 cpu_buffer 的开头部分
    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder='big')

    # 将序列化后的对象（enc）拷贝到 cpu_buffer 中，紧跟在编码的大小信息后面
    cpu_buffer[0:SIZE_STORAGE_BYTES] = torch.ByteTensor(list(size_bytes))
    cpu_buffer[SIZE_STORAGE_BYTES: enc_size + SIZE_STORAGE_BYTES] = torch.ByteTensor(list(enc))

    start = rank * max_size
    size = enc_size + SIZE_STORAGE_BYTES
    buffer[start: start + size].copy_(cpu_buffer[:size])

    # 将准备好的数据从 cpu_buffer 拷贝到 GPU 的 buffer，然后使用 all_reduce 操作。all_reduce 在这里可能是用于确保所有 GPU 都有完整的数据集合
    all_reduce(buffer, group=group)

    try:
        # 在 all_reduce 完成后，每个进程（或 GPU）将包含所有进程序列化数据的完整副本。接下来，遍历每个进程的数据，解析出对象的大小，然后反序列化（通过 pickle.loads）对象
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = int.from_bytes(out_buffer[0:SIZE_STORAGE_BYTES], byteorder='big')
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES: size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list requires all '
            'workers to enter the function together, so this error usually indicates '
            'that the workers have fallen out of sync somehow. Workers can fall out of '
            'sync if one of them runs out of memory, or if there are other conditions '
            'in your training script that can cause one worker to finish an epoch '
            'while other workers are still iterating over their portions of the data.'
        )

""" 
.to_bytes() 是 Python 中的一个方法，它用于将一个整数转换成字节表示

pickle.loads() 是 Python pickle 模块的一个函数，用于将序列化的对象（即通过 pickle.dump() 或 pickle.dumps() 函数序列化并存储为字节流的对象）反序列化回其原始形式。这个函数的参数是包含有序列化数据的字节对象，返回值是原始的 Python 对象

int.from_bytes() 将字节对象转换为整数的类方法。它允许指定字节顺序（即大端或小端）和整数是否应被视为有符号整数

byteorder 参数指定了字节序。在二进制数据中，字节序可以是大端（'big'）或小端（'little'）。大端字节序意味着最高有效位存储在最低的字节地址（即“大头”在前），而小端字节序则相反，最低有效位存储在最低的字节地址（即“小头”在前）

.bytes() 是 Python 中的一个内置函数，用于生成一个新的字节对象。这是不可变的序列，包含范围在 0 <= x < 256 的整数。bytes() 可以接受多种类型的输入，如字符串（需指定编码）、整数（创建指定长度的零填充字节对象）或可迭代对象（包含整数）

.tolist() 是 numpy 数组（ndarray）的一个方法，用于将数组转换为等价的嵌套 Python 列表结构。这在将数据从 numpy 数组格式转换为 Python 标准数据结构时非常有用

pickle.UnpicklingError 是 pickle 模块定义的一个异常，它在反序列化（即使用 pickle.loads() 或相似函数）过程中，如果遇到一个无法识别或损坏的序列化对象时抛出。这是 pickle 操作中可能遇到的多种错误之一，表明反序列化过程失败
"""