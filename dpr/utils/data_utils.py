#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for general purpose data processing
"""
import json
import logging
import pickle
import random

import itertools
import math

import hydra
import jsonlines
import torch
from omegaconf import DictConfig
from torch import Tensor as T
from typing import List, Iterator, Callable, Tuple

logger = logging.getLogger()


def read_serialized_data_from_files(paths: List[str]) -> List:
    # 从一个或多个文件中读取序列化数据，并将这些数据聚合成一个列表返回。
    results = [] # 存储从所有文件中读取的数据
    for i, path in enumerate(paths):
        # 使用 with 语句以二进制读取模式（"rb"）打开文件。这种模式对于读取序列化数据（如使用 pickle 模块序列化的数据）是必需的。with 语句确保文件在操作完成后能够正确关闭
        with open(path, "rb") as reader:
            # 使用 logger.info 记录开始读取文件的信息。这有助于跟踪程序的执行状态和诊断问题
            logger.info("Reading file %s", path)
            data = pickle.load(reader)
            results.extend(data)
            # 每读取一个文件并将其数据添加到结果列表后，记录当前结果列表的大小
            logger.info("Aggregated data size: {}".format(len(results)))
    # 当所有文件都被读取并聚合后，记录最终的总数据大小
    logger.info("Total data size: {}".format(len(results)))
    return results


def read_data_from_json_files(paths: List[str]) -> List:
    # 从一系列JSON文件中读取数据并将其聚合成一个列表
    # 该函数接受一个包含文件路径的列表paths作为输入，并返回一个包含所有文件数据的列表
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            # 使用with语句安全地打开文件。这里指定以读取模式（"r"）打开文件，并使用utf-8编码来确保正确处理文件中的字符
            logger.info("Reading file %s" % path) # 在日志中记录当前正在读取的文件路径。这有助于调试和跟踪函数的执行情况
            data = json.load(f)
            results.extend(data) # 将读取的数据data添加到results列表中。extend方法用于将一个列表的所有元素添加到另一个列表中，这里用于聚合来自不同文件的数据
            logger.info("Aggregated data size: {}".format(len(results))) # 在日志中记录到目前为止聚合的数据总量。这有助于了解数据聚合的进度和最终大小
    return results


def read_data_from_jsonl_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        logger.info("Reading file %s" % path)
        with jsonlines.open(path, mode="r") as jsonl_reader:
            data = [r for r in jsonl_reader]
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    return results


def normalize_question(question: str) -> str:
    question = question.replace("’", "'") # 替换字符串中的某些字符或子串。在这个例子中，它将所有的“’”（右单引号或弯引号）替换为标准的单引号“'”（直单引号）
    return question


class Tensorizer(object):
    # 该类是一个抽象基类，用于将文本转换成适合输入到模型的张量形式，以及实现与文本和张量转换相关的其他一些实用方法
    # 所有这些方法都使用 raise NotImplementedError，这意味着 Tensorizer 类的目的是作为一个接口或基类使用。具体的实现应该在继承了 Tensorizer 的子类中提供
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        # 将给定的文本（和可选的标题）转换成模型可处理的张量格式
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        # 返回用于分隔文本对（如问题-答案对或文本-标题对）的分隔符对应的标识符序列
        raise NotImplementedError

    def get_pad_id(self) -> int:
        # 返回用于填充序列的填充标识符（pad token ID）
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        # 根据给定的令牌张量生成注意力掩码，用于模型中忽略填充令牌
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        # 判断给定的令牌ID是否表示一个子词
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        # 将一系列令牌ID转换回文本字符串
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        # 设置是否应将序列填充到最大长度
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        #  根据给定的令牌获取其ID
        raise NotImplementedError


class RepTokenSelector(object):
    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):  # 根据输入的ID和一个Tensorizer对象来确定令牌的位置
        raise NotImplementedError


class RepStaticPosTokenSelector(RepTokenSelector):
    def __init__(self, static_position: int = 0):
        # static_position 参数允许在创建类的实例时指定这个静态位置，默认值为 0。这意味着如果没有特别指定，get_positions 将返回 0，这通常对应于输入序列的第一个令牌
        self.static_position = static_position

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        # 不论输入数据如何，总是返回一个静态的、预定义的位置 self.static_position
        return self.static_position


class RepSpecificTokenSelector(RepTokenSelector):
    def __init__(self, token: str = "[CLS]"):
        self.token = token
        self.token_id = None

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        if not self.token_id:
            self.token_id = tenzorizer.get_token_id(self.token)
        token_indexes = (input_ids == self.token_id).nonzero()
        # check if all samples in input_ids has index presence and out a default value otherwise
        bsz = input_ids.size(0)
        if bsz == token_indexes.size(0):
            return token_indexes

        token_indexes_result = []
        found_idx_cnt = 0
        for i in range(bsz):
            if found_idx_cnt < token_indexes.size(0) and token_indexes[found_idx_cnt][0] == i:
                # this samples has the special token
                token_indexes_result.append(token_indexes[found_idx_cnt])
                found_idx_cnt += 1
            else:
                logger.warning("missing special token %s", input_ids[i])

                token_indexes_result.append(
                    torch.tensor([i, 0]).to(input_ids.device)
                )  # setting 0-th token, i.e. CLS for BERT as the special one
        token_indexes_result = torch.stack(token_indexes_result, dim=0)
        return token_indexes_result


DEFAULT_SELECTOR = RepStaticPosTokenSelector()


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        selector: DictConfig = None,
        special_token: str = None,
        shuffle_positives: bool = False,
        query_special_suffix: str = None,
        encoder_type: str = None,
    ):
        """ 
        selector: 一个DictConfig对象（通常由Hydra配置管理库提供），用于选择数据。如果提供了selector，则使用hydra.utils.instantiate(selector)实例化一个数据选择器；如果未提供，则使用默认选择器DEFAULT_SELECTOR。
        special_token: 特殊标记字符串，可以添加到数据中。
        shuffle_positives: 布尔值，指示是否在使用数据前对正样本进行随机排序。
        query_special_suffix: 查询特殊后缀，如果设置了这个参数，则所有的查询字符串在处理时都会附加这个后缀。
        encoder_type: 编码器类型，指定如何编码数据。
        self.data = []: 初始化一个空列表用于存储数据集中的数据
        """
        if selector:
            self.selector = hydra.utils.instantiate(selector)
        else:
            self.selector = DEFAULT_SELECTOR
        self.special_token = special_token
        self.encoder_type = encoder_type
        self.shuffle_positives = shuffle_positives
        self.query_special_suffix = query_special_suffix
        self.data = []

    def load_data(self, start_pos: int = -1, end_pos: int = -1):
        # 声明了一个用于加载数据的方法。该方法接受start_pos和end_pos两个参数，分别表示数据加载的起始和结束位置
        raise NotImplementedError

    def calc_total_data_len(self):
        # 声明了一个用于计算数据总长度的方法
        raise NotImplementedError

    def __len__(self):
        # 实现了特殊方法__len__，使得可以使用len(dataset)直接获取数据集的长度，这里返回的是self.data列表的长度
        return len(self.data)

    def __getitem__(self, index):
        # 可以使用索引直接从数据集中获取数据项，例如dataset[0]
        raise NotImplementedError

    def _process_query(self, query: str):
        # as of now, always normalize query
        # 定义了一个辅助方法用于处理查询字符串。首先，它使用normalize_question函数标准化查询字符串。如果设置了query_special_suffix并且查询不以这个后缀结尾，则会将这个后缀添加到查询字符串的末尾
        query = normalize_question(query)
        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix

        return query


# TODO: to be fully replaced with LocalSharded{...}. Keeping it only for old results reproduction compatibility
class ShardedDataIterator(object):
    """
    旨在支持PyTorch的分布式数据并行(DDP)模式下，高效地迭代数据集的分片。在DDP模式下，每个节点处理数据的一个子集，ShardedDataIterator使这个过程更加便捷
    """

    def __init__(
        self,
        dataset: Dataset,
        shard_id: int = 0,
        num_shards: int = 1,
        batch_size: int = 1,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        strict_batch_size: bool = False,
    ):
        """
        初始化数据集迭代器，接收数据集(dataset)和分片相关的参数，如分片ID(shard_id)、分片总数(num_shards)、批大小(batch_size)等，以及是否打乱(shuffle)和打乱种子(shuffle_seed)。
        self.iteration用于跟踪当前分片内的迭代状态。
        self.shard_start_idx和self.shard_end_idx确定每个分片的起始和结束索引，而self.max_iterations指示在当前分片上可以进行多少次迭代 
        """

        self.dataset = dataset
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size
        self.shard_start_idx = -1
        self.shard_end_idx = -1
        self.max_iterations = 0

    def calculate_shards(self):
        # 计算每个分片的起始和结束索引，以及该分片内的最大迭代次数。如果设置了strict_batch_size，则每次迭代的批大小将严格等于指定的batch_size
        logger.info("Calculating shard positions") # 标识开始计算分片的位置
        # 确保分片数量至少为1
        shards_num = max(self.num_shards, 1)
        # 确保分片ID至少为0
        shard_id = max(self.shard_id, 0) 

        # 计算数据集的总大小，即数据集中的样本总数
        total_size = self.dataset.calc_total_data_len()
        # 计算每个分片应有的样本数。使用 math.ceil 确保即使无法均匀分配时，每个分片也能尽可能地包含足够的样本
        samples_per_shard = math.ceil(total_size / shards_num)

        # 计算当前分片的起始索引。通过当前分片ID乘以每个分片的样本数得到
        self.shard_start_idx = shard_id * samples_per_shard
        # 计算当前分片的结束索引。取 shard_start_idx + samples_per_shard 和数据集总大小的最小值，以确保不会超出数据集的边界
        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        if self.strict_batch_size:
            # 表示每次迭代的批大小应严格等于指定的 self.batch_size，使用 math.ceil(samples_per_shard / self.batch_size) 来确保即使无法均匀分配，每个批次也至少有指定的批大小
            self.max_iterations = math.ceil(samples_per_shard / self.batch_size)
        else:
            # 使用整数除法 int(samples_per_shard / self.batch_size)，这可能导致最后一个批次的大小小于指定的批大小
            self.max_iterations = int(samples_per_shard / self.batch_size)

        # 记录当前分片的一些详细信息，包括每个分片的样本数、起始索引、结束索引以及最大迭代次数
        logger.info(
            "samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d",
            samples_per_shard,
            self.shard_start_idx,
            self.shard_end_idx,
            self.max_iterations,
        )

    def load_data(self):
        # 调用calculate_shards方法来计算分片，然后加载数据集。这个方法可能需要在子类中实现具体的数据加载逻辑
        self.calculate_shards()
        self.dataset.load_data()
        logger.info("Sharded dataset data %d", len(self.dataset))

    def total_data_len(self) -> int:
        # 返回数据集的总长度
        return len(self.dataset)

    def iterations_num(self) -> int:
        # 返回当前分片剩余的迭代次数
        return self.max_iterations - self.iteration

    def max_iterations_num(self) -> int:
        # 返回当前分片的最大迭代次数
        return self.max_iterations

    def get_iteration(self) -> int:
        # 返回当前的迭代次数
        return self.iteration

    def apply(self, visitor_func: Callable):
        # 对数据集中的每个样本应用一个函数(visitor_func)
        for sample in self.dataset:
            visitor_func(sample)

    def get_shard_indices(self, epoch: int):
        # 根据当前的epoch和是否打乱(shuffle)，获取当前分片的索引列表
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)
        shard_indices = indices[self.shard_start_idx : self.shard_end_idx]
        return shard_indices

    # TODO: merge with iterate_ds_sampled_data
    def iterate_ds_data(self, epoch: int = 0) -> Iterator[List]:
        # 根据分片中的数据和批大小(batch_size)来迭代数据，对于每个epoch，它生成一个批次的数据项, 训练过程中需要遍历整个分片的场景
        # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations
        max_iterations = self.max_iterations - self.iteration
        shard_indices = self.get_shard_indices(epoch) # 根据当前epoch，使用get_shard_indices方法计算分片的索引列表

        for i in range(self.iteration * self.batch_size, len(shard_indices), self.batch_size):
            # 遍历这些索引并每次前进batch_size个元素，来组成一个批次的数据项索引(items_idxs)
            items_idxs = shard_indices[i : i + self.batch_size]
            if self.strict_batch_size and len(items_idxs) < self.batch_size:
                # 如果开启了strict_batch_size且最后一个批次的数据量不足，它会从分片的开始处补充数据项以确保每个批次大小恒定
                logger.debug("Extending batch to max size")
                items_idxs.extend(shard_indices[0 : self.batch_size - len(items)])
            self.iteration += 1
            items = [self.dataset[idx] for idx in items_idxs] # 使用列表推导式，通过索引从数据集中获取对应的数据项，然后将这些数据项作为一个列表返回
            yield items

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        # 当迭代完成或某些分片比其他分片早完成时，它会重复返回第一个批次的数据，直到所有分片都完成迭代
        while self.iteration < max_iterations:
            logger.debug("Fulfilling non complete shard=".format(self.shard_id))
            self.iteration += 1
            items_idxs = shard_indices[0 : self.batch_size]
            items = [self.dataset[idx] for idx in items_idxs]
            yield items

        # 每次迭代完成后，记录日志信息并重置迭代计数器
        logger.info("Finished iterating, iteration={}, shard={}".format(self.iteration, self.shard_id))
        # reset the iteration status
        self.iteration = 0

    def iterate_ds_sampled_data(self, num_iterations: int, epoch: int = 0) -> Iterator[List]:
        # 这个方法允许在指定num_iterations（迭代次数）的情况下迭代数据，不依赖于分片的实际大小, 需要固定迭代次数的评估或测试场景
        self.iteration = 0
        shard_indices = self.get_shard_indices(epoch) # 根据当前epoch，计算分片的索引列表
        cycle_it = itertools.cycle(shard_indices) # 使用itertools.cycle创建一个循环迭代器，允许无限次序地遍历分片的索引
        for i in range(num_iterations):
            # 在每次迭代中，通过next(cycle_it)从循环迭代器中获取batch_size个数据项索引，然后从数据集中获取对应的数据项
            items_idxs = [next(cycle_it) for _ in range(self.batch_size)]
            self.iteration += 1
            items = [self.dataset[idx] for idx in items_idxs]
            yield items

        # 每次迭代完成后，记录日志信息。需要注意的是，这里提到了“TODO: reset the iteration status?”，暗示在方法结束时是否应该重置迭代计数器仍是一个待决策的问题
        logger.info("Finished iterating, iteration={}, shard={}".format(self.iteration, self.shard_id))
        # TODO: reset the iteration status?
        self.iteration = 0

    def get_dataset(self) -> Dataset:
        # 返回初始化时提供的数据集
        return self.dataset


class LocalShardedDataIterator(ShardedDataIterator):
    # 重写了父类中的某些方法以更适合本地处理分片数据的需求，特别是为了减少内存占用
    # uses only one shard after the initial dataset load to reduce memory footprint
    def load_data(self):
        # 此方法的目的是加载数据集，但与ShardedDataIterator中的实现不同，LocalShardedDataIterator仅加载其分配到的那部分分片数据。这种方式有助于在处理大型数据集时减少每个进程的内存占用
        self.calculate_shards() # 计算当前进程/节点应处理的分片的起始和结束索引
        self.dataset.load_data(start_pos=self.shard_start_idx, end_pos=self.shard_end_idx) # 使用这些索引调用数据集的load_data方法，这意味着只有分配给当前分片的数据被加载，而不是整个数据集。这通常需要Dataset类中的load_data方法能够接受开始和结束位置作为参数
        logger.info("Sharded dataset data %d", len(self.dataset)) # 记录加载的数据量，以便于调试和监测

    def get_shard_indices(self, epoch: int):
        # 获取当前分片的数据索引列表，这一步骤与父类略有不同，因为它假定数据已经被限制在了当前分片内，因此直接返回当前已加载数据的索引列表
        indices = list(range(len(self.dataset))) # 创建一个包含当前已加载数据索引的列表
        if self.shuffle:
            # 如果需要打乱数据，它会使用一个基于epoch和shuffle_seed的随机数生成器来打乱这些索引。这保证了即使在多个训练周期中或在从失败/停止的迭代恢复时，也能够使用相同的数据打乱顺序
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)
        shard_indices = indices
        # 返回打乱后（如果启用了打乱）的索引列表。由于数据已经预先被限制在当前分片内，这个列表实际上包含了当前进程/节点处理的所有数据的索引
        return shard_indices


class MultiSetDataIterator(object):
    """
    从多个数据源（每个都是一个 ShardedDataIterator 实例）迭代数据而设计的。它对这些数据源进行统一管理，以便所有来自同一个批次的样本都来自同一个数据集
    datasets: 包含 ShardedDataIterator 实例的列表，每个实例代表一个数据源。
    shuffle_seed: 用于确定数据混洗顺序的种子。
    shuffle: 一个布尔值，指示是否在每个周期（epoch）开始时混洗数据源的顺序。
    sampling_rates: 一个列表，包含每个数据集的采样率，用于控制从每个数据集中抽取的样本数量。
    rank: 用于数据混洗的随机种子计算中的排名因子，通常与分布式训练中的进程标识符相关联。
    """

    def __init__(
        self,
        datasets: List[ShardedDataIterator],
        shuffle_seed: int = 0,
        shuffle=True,
        sampling_rates: List = [],
        rank: int = 0,
    ):
        # randomized data loading to avoid file system congestion
        # 复制数据集列表并根据 rank 混洗这个列表，以避免文件系统拥塞
        ds_list_copy = [ds for ds in datasets]
        rnd = random.Random(rank)
        rnd.shuffle(ds_list_copy)
        [ds.load_data() for ds in ds_list_copy] # 调用每个数据源的 load_data 方法来加载数据

        self.iterables = datasets
        # 计算所有数据集的总数据量和每个数据集的最大迭代次数
        data_lengths = [it.total_data_len() for it in datasets]
        self.total_data = sum(data_lengths)
        # 记录相关信息，包括数据集大小、总数据量、采样率和最大迭代次数
        logger.info("rank=%d; Multi set data sizes %s", rank, data_lengths)
        logger.info("rank=%d; Multi set total data %s", rank, self.total_data)
        logger.info("rank=%d; Multi set sampling_rates %s", rank, sampling_rates)
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.iteration = 0
        self.rank = rank

        if sampling_rates:
            # 如果提供了 sampling_rates 参数（一个包含每个数据集采样率的列表），代码会遍历 datasets 列表，并对每个数据集使用其对应的采样率来计算最大迭代次数。采样率乘以每个数据集的最大迭代次数（通过调用数据集的 max_iterations_num 方法获得）给出了调整后的最大迭代次数
            self.max_its_pr_ds = [int(ds.max_iterations_num() * sampling_rates[i]) for i, ds in enumerate(datasets)]
        else:
            # 如果没有提供 sampling_rates，则对于每个数据集，其最大迭代次数将直接等于该数据集能提供的最大迭代次数，不进行任何调整
            self.max_its_pr_ds = [ds.max_iterations_num() for ds in datasets]

        self.max_iterations = sum(self.max_its_pr_ds) # 计算所有数据集的最大迭代次数之和，得到 self.max_iterations。这表示如果要从每个数据集中均匀抽取样本，迭代器能够提供的总迭代次数
        # 使用 logger.info 方法记录每个数据集的最大迭代次数和总最大迭代次数，以及当前的 rank。这有助于调试和了解迭代器如何在不同的训练阶段分配数据
        logger.info("rank=%d; Multi set max_iterations per dataset %s", rank, self.max_its_pr_ds)
        logger.info("rank=%d; Multi set max_iterations %d", rank, self.max_iterations)

    def total_data_len(self) -> int:
        # 返回所有数据集的总数据长度
        return self.total_data

    def get_max_iterations(self):
        # 返回所有数据集可以提供的最大迭代次数的总和
        return self.max_iterations

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[Tuple[List, int]]:
        # 在给定的周期（epoch）迭代多个数据集的数据，并确保可以从每个数据集中顺序获取数据，直到达到每个数据集的预定迭代次数。方法的实现考虑了数据混洗和跨多个数据源的均匀采样

        # 记录迭代开始的日志信息，包括当前的 rank 和每个数据集当前的迭代次数
        logger.info("rank=%d; Iteration start", self.rank)
        logger.info(
            "rank=%d; Multi set iteration: iteration ptr per set: %s",
            self.rank,
            [it.get_iteration() for it in self.iterables],
        )

        data_src_indices = []   # 存储将要从中采样数据的数据集索引，每个数据集的索引会根据其在 self.max_its_pr_ds 中定义的迭代次数重复出现，确保数据均匀采样
        iterators = []
        for source, src_its in enumerate(self.max_its_pr_ds):
            logger.info(
                "rank=%d; Multi set iteration: source %d, batches to be taken: %s",
                self.rank,
                source,
                src_its,
            )
            data_src_indices.extend([source] * src_its)

            # 对于每个数据集，调用其 iterate_ds_sampled_data 方法来创建一个迭代器，这些迭代器存储在 iterators 列表中。每个迭代器负责迭代特定数据集的数据
            iterators.append(self.iterables[source].iterate_ds_sampled_data(src_its, epoch=epoch))

        if self.shuffle:
            # 如果启用了混洗（self.shuffle 为 True），使用特定的种子（self.shuffle_seed + epoch）来混洗 data_src_indices 列表。这样做旨在确保即使在训练中断后重新开始时，同一周期的数据混洗顺序也保持一致
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(data_src_indices)

        logger.info("rank=%d; data_src_indices len=%d", self.rank, len(data_src_indices))
        # 通过枚举 data_src_indices，使用每个索引来选择对应的迭代器，并从中获取下一个数据项
        for i, source_idx in enumerate(data_src_indices):
            it = iterators[source_idx]
            next_item = next(it, None)
            # 如果从迭代器中成功获取到数据项，增加 self.iteration 计数，并使用 yield 语句返回这个数据项及其来源数据集的索引
            if next_item is not None:
                self.iteration += 1
                yield (next_item, source_idx)
            else:
                # 如果某个数据源的迭代器返回 None（表示没有更多数据可供迭代），记录一个警告日志
                logger.warning("rank=%d; Next item in the source %s is None", self.rank, source_idx)

        logger.info("rank=%d; last iteration %d", self.rank, self.iteration)

        # 在所有数据都被迭代完毕后，记录迭代结束的信息，并尝试从每个迭代器中获取一次数据，以确保它们处于完成状态
        logger.info(
            "rank=%d; Multi set iteration finished: iteration per set: %s",
            self.rank,
            [it.iteration for it in self.iterables],
        )
        [next(it, None) for it in iterators]

        # TODO: clear iterators in some non-hacky way
        # 重置每个数据集迭代器的迭代次数，并记录最终的迭代状态
        for it in self.iterables:
            it.iteration = 0
        logger.info(
            "rank=%d; Multi set iteration finished after next: iteration per set: %s",
            self.rank,
            [it.iteration for it in self.iterables],
        )
        # reset the iteration status
        self.iteration = 0 # 将 self.iteration 重置为0，为下一个周期的迭代准备。

    def get_iteration(self) -> int:
        # 返回当前迭代的次数
        return self.iteration

    # 提供了获取单个数据集或所有数据集实例的方式
    def get_dataset(self, ds_id: int) -> Dataset:
        return self.iterables[ds_id].get_dataset()

    def get_datasets(self) -> List[Dataset]:
        return [it.get_dataset() for it in self.iterables]
