#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Set of utilities for the Reader model related data processing tasks
"""

import collections
import glob
import json
import logging
import math
import multiprocessing
import os
import pickle
from functools import partial
from typing import Tuple, List, Dict, Iterable, Optional

import torch
from torch import Tensor as T
from tqdm import tqdm

from dpr.utils.data_utils import (
    Tensorizer,
    read_serialized_data_from_files,
    read_data_from_json_files,
    Dataset as DprDataset,
)

logger = logging.getLogger()


class ReaderPassage(object):
    """
    在生成阅读理解模型输入之前收集和缓存与问答段落相关的所有属性
    id：段落的唯一标识符。
    passage_text：段落的文本内容。
    title：段落的标题。
    score：段落的得分或相关性评分。
    has_answer：一个布尔值，指示该段落是否包含问题的答案。
    passage_token_ids：段落文本的token ID列表，用于模型处理。
    passage_offset：实际段落（不是问题或可能的标题）在sequence_ids中的偏移量。
    answers_spans：答案在段落中的跨度列表。
    sequence_ids：包括问题和段落文本的整个序列的token ID列表。
    """

    def __init__(
        self,
        id=None,
        text: str = None,
        title: str = None,
        score=None,
        has_answer: bool = None,
    ):
        self.id = id
        # string passage representations
        self.passage_text = text
        self.title = title
        self.score = score
        self.has_answer = has_answer
        self.passage_token_ids = None
        # offset of the actual passage (i.e. not a question or may be title) in the sequence_ids
        self.passage_offset = None
        self.answers_spans = None
        # passage token ids
        self.sequence_ids = None

    def on_serialize(self):
        # 将sequence_ids转换为NumPy数组格式，以便序列化，并清除passage_text、title和passage_token_ids属性，因为序列化通常是为了减少存储空间和传输时的带宽消耗
        # store only final sequence_ids and the ctx offset
        self.sequence_ids = self.sequence_ids.numpy()
        self.passage_text = None
        self.title = None
        self.passage_token_ids = None

    def on_deserialize(self):
        # 将sequence_ids从NumPy数组格式转换回PyTorch张量，以便于模型处理。其他文本属性在序列化过程中已被清除，因此不需要恢复。
        self.sequence_ids = torch.tensor(self.sequence_ids)


class ReaderSample(object):
    """
    在单个问题下收集所有问答（Q&A）通道的数据
    question: 表示问题的字符串。
    answers: 一个列表，包含问题的答案。
    positive_passages: 一个 ReaderPassage 对象的列表，默认为空列表。这些是与问题相关联的正面通道。
    negative_passages: 一个 ReaderPassage 对象的列表，默认为空列表。这些是与问题不相关的负面通道。
    passages: 一个 ReaderPassage 对象的列表，默认为空列表。这个列表用于存放与问题相关的其他通道。
    """

    def __init__(
        self,
        question: str,
        answers: List,
        positive_passages: List[ReaderPassage] = [],
        negative_passages: List[ReaderPassage] = [],
        passages: List[ReaderPassage] = [],
    ):
        self.question = question
        self.answers = answers
        self.positive_passages = positive_passages
        self.negative_passages = negative_passages
        self.passages = passages

    def on_serialize(self):
        # 在序列化（例如，准备将对象存储或发送）过程中对所有通道（正面、负面和其他通道）进行操作
        for passage in self.passages + self.positive_passages + self.negative_passages:
            passage.on_serialize()

    def on_deserialize(self):
        # 在反序列化（例如，从存储中恢复或接收后处理）过程中对所有通道进行操作
        for passage in self.passages + self.positive_passages + self.negative_passages:
            passage.on_deserialize()


class ExtractiveReaderDataset(torch.utils.data.Dataset):
    # 处理和加载用于抽取式阅读理解模型训练和评估的数据集
    def __init__(
        self,
        files: str,
        is_train: bool,
        gold_passages_src: str,
        tensorizer: Tensorizer,
        run_preprocessing: bool,
        num_workers: int,
    ):
        """ 
        files：数据文件的路径或模式。
        is_train：一个布尔值，指示是否为训练数据集。
        gold_passages_src：含有正确答案的金标准数据源的路径。
        tensorizer：Tensorizer对象，用于数据的预处理和转换。
        run_preprocessing：一个布尔值，指示是否在加载数据前运行预处理步骤。
        num_workers：用于数据预处理的工作进程数量。
        """
        self.files = files
        self.data = []
        self.is_train = is_train
        self.gold_passages_src = gold_passages_src
        self.tensorizer = tensorizer
        self.run_preprocessing = run_preprocessing
        self.num_workers = num_workers

    def __getitem__(self, index):
        # 实现数据集的索引操作，允许直接通过索引获取数据集中的单个样本
        return self.data[index]

    def __len__(self):
        # 返回数据集中样本的总数量
        return len(self.data)

    def calc_total_data_len(self):
        # 计算并返回数据集中样本的总数量。如果数据尚未加载，则首先调用load_data方法加载数据。
        if not self.data:
            self.load_data()
        return len(self.data)

    def load_data(
        self,
    ):
        # 负责加载数据。如果数据已经加载，方法将直接返回。方法首先寻找数据文件，然后检查数据是否已经预处理。如果已预处理，直接加载预处理后的文件；如果未预处理且run_preprocessing为True，则进行预处理并加载结果
        if self.data:
            return

        data_files = glob.glob(self.files)
        logger.info("Data files: %s", data_files)
        if not data_files:
            raise RuntimeError("No Data files found")
        preprocessed_data_files = self._get_preprocessed_files(data_files)
        self.data = read_serialized_data_from_files(preprocessed_data_files)

    def _get_preprocessed_files(
        self,
        data_files: List,
    ):
        # 找到或生成预处理（序列化）后的数据文件，以供抽取式问答模型训练使用

        # 检查传入的data_files列表中是否有以.pkl结尾的文件（已序列化的文件）
        serialized_files = [file for file in data_files if file.endswith(".pkl")]
        if serialized_files:
            # 如果找到这样的文件，方法将直接返回这些文件，避免重复预处理
            return serialized_files
        # 断言确保只提供了一个数据源文件进行预处理，因为当前实现只支持处理单个文件
        assert len(data_files) == 1, "Only 1 source file pre-processing is supported."

        # 在相同目录下查找可能已经序列化和缓存的文件
        def _find_cached_files(path: str):
            # 该函数根据原始数据文件的路径生成可能的序列化文件名模式，然后返回匹配该模式的所有文件
            dir_path, base_name = os.path.split(path)
            base_name = base_name.replace(".json", "")
            out_file_prefix = os.path.join(dir_path, base_name)
            out_file_pattern = out_file_prefix + "*.pkl"
            return glob.glob(out_file_pattern), out_file_prefix

        serialized_files, out_file_prefix = _find_cached_files(data_files[0])
        if serialized_files:
            logger.info("Found preprocessed files. %s", serialized_files)
            return serialized_files

        # 如果没有找到已序列化的文件，会记录一条日志消息，表明将开始数据预处理过程
        logger.info("Data are not preprocessed for reader training. Start pre-processing ...")

        # start pre-processing and save results
        def _run_preprocessing(tensorizer: Tensorizer):
            # 实际执行数据预处理
            # temporarily disable auto-padding to save disk space usage of serialized files
            tensorizer.set_pad_to_max(False) # 禁用自动填充，以减少序列化文件的磁盘空间使用、
            # 将原始数据文件转换为序列化格式，同时考虑到是否为训练集、金标准通道源文件
            serialized_files = convert_retriever_results(
                self.is_train,
                data_files[0],
                out_file_prefix,
                self.gold_passages_src,
                self.tensorizer,
                num_workers=self.num_workers,
            ) 
            tensorizer.set_pad_to_max(True) # 恢复自动填充设置
            return serialized_files

        if self.run_preprocessing:
            # 如果run_preprocessing标志设置为True，则调用_run_preprocessing函数执行预处理并返回生成的序列化文件
            serialized_files = _run_preprocessing(self.tensorizer)
            # TODO: check if pytorch process group is initialized
            # torch.distributed.barrier()
        else:
            # 如果不执行预处理，则再次尝试查找缓存的序列化文件
            # torch.distributed.barrier()
            serialized_files = _find_cached_files(data_files[0])
        # 返回找到或生成的序列化文件列表，供数据集加载和处理使用
        return serialized_files


SpanPrediction = collections.namedtuple(
    "SpanPrediction",
    [
        "prediction_text",
        "span_score",
        "relevance_score",
        "passage_index",
        "passage_token_ids",
    ],
)

# configuration for reader model passage selection
ReaderPreprocessingCfg = collections.namedtuple(
    "ReaderPreprocessingCfg",
    [
        "use_tailing_sep",
        "skip_no_positves",
        "include_gold_passage",
        "gold_page_only_positives",
        "max_positives",
        "max_negatives",
        "min_negatives",
        "max_retriever_passages",
    ],
)

DEFAULT_PREPROCESSING_CFG_TRAIN = ReaderPreprocessingCfg(
    use_tailing_sep=False,
    skip_no_positves=True,
    include_gold_passage=False,  # True - for speech Q&A
    gold_page_only_positives=True,
    max_positives=20,
    max_negatives=50,
    min_negatives=150,
    max_retriever_passages=200,
)

DEFAULT_EVAL_PASSAGES = 100


def preprocess_retriever_data(
    samples: List[Dict],
    gold_info_file: Optional[str],
    tensorizer: Tensorizer,
    cfg: ReaderPreprocessingCfg = DEFAULT_PREPROCESSING_CFG_TRAIN,
    is_train_set: bool = True,
) -> Iterable[ReaderSample]:
    """
    将检索器的结果转换为阅读器模型的训练数据
    处理原始样本集，为每个样本生成一个适用于阅读器模型的ReaderSample对象
    
    samples：检索器的json文件结果中的样本列表。
    gold_info_file：'金标准通道 & 问题'文件的可选路径。提供此文件可以获得NQ数据集的最佳结果。
    tensorizer：Tensorizer对象，用于将文本转换为模型输入张量。
    cfg：带有正面和负面通道选择参数的ReaderPreprocessingCfg对象。
    is_train_set：指示数据是否应作为训练集进行处理的布尔值。
    """
    # 获取用于分隔问题和通道文本的分隔符张量
    sep_tensor = tensorizer.get_pair_separator_ids()  # separator can be a multi token
    # 如果提供了金标准文件，则解析该文件以获得金标准通道映射和规范化问题
    gold_passage_map, canonical_questions = _get_gold_ctx_dict(gold_info_file) if gold_info_file else ({}, {})

    no_positive_passages = 0
    positives_from_gold = 0

    # 为阅读理解任务中的一个样本生成所需的张量ID，并对样本进行适当的处理
    def create_reader_sample_ids(sample: ReaderPassage, question: str):
        """ 
        sample: ReaderPassage 对象，包含标题 (title)、通道文本 (passage_text)、以及可能的答案位置 (answers_spans)。
        question: 与 sample 相关的问题文本。
        """
        # 将问题和标题文本转换成张量（通常是一个整数列表，代表文本中每个词的ID）。这里设置 add_special_tokens=True 来添加特殊令牌（如BERT的[CLS]和[SEP]），这对于某些模型是必需的
        question_and_title = tensorizer.text_to_tensor(sample.title, title=question, add_special_tokens=True)
        # 如果 sample 中的 passage_token_ids 尚未设置，这段代码将通道文本转换为张量，并且不添加特殊令牌（因为这些将会在问题和标题张量中处理）。
        if sample.passage_token_ids is None:
            sample.passage_token_ids = tensorizer.text_to_tensor(sample.passage_text, add_special_tokens=False)

        # 使用 _concat_pair 函数将问题和标题的张量与通道文本的张量拼接起来。如果配置中指定了使用尾部分隔符 (cfg.use_tailing_sep)，则会添加一个尾部分隔符（sep_tensor）
        all_concatenated, shift = _concat_pair(
            question_and_title,
            sample.passage_token_ids,
            tailing_sep=sep_tensor if cfg.use_tailing_sep else None,
        )

        # 更新样本对象，设置拼接后的序列ID（sequence_ids）和通道文本在拼接序列中的起始位置（passage_offset）。shift 变量表示通道文本在拼接序列中的偏移量
        sample.sequence_ids = all_concatenated
        sample.passage_offset = shift
        # 使用断言确保偏移量 (shift) 大于1，这是合理的，因为在序列的开头至少应该有一个问题标记和一个分隔符
        assert shift > 1
        # 如果样本包含答案（sample.has_answer 为真），并且正在处理的是训练集（is_train_set 为真），则更新答案跨度（answers_spans），将每个答案跨度根据通道文本的偏移量调整
        if sample.has_answer and is_train_set:
            sample.answers_spans = [(span[0] + shift, span[1] + shift) for span in sample.answers_spans]
        return sample # 返回处理后的样本对象

    for sample in samples:
        # 对于给定的每个样本，提取问题文本，并尝试使用规范化问题（如果存在）
        question = sample["question"]
        question_txt = sample["query_text"] if "query_text" in sample else question

        if canonical_questions and question_txt in canonical_questions:
            question_txt = canonical_questions[question_txt]

        # 根据配置选择正面和负面的通道
        positive_passages, negative_passages = _select_reader_passages(
            sample,
            question_txt,
            tensorizer,
            gold_passage_map,
            cfg.gold_page_only_positives,
            cfg.max_positives,
            cfg.max_negatives,
            cfg.min_negatives,
            cfg.max_retriever_passages,
            cfg.include_gold_passage,
            is_train_set,
        )
        # 对于每个选定的通道（正面和负面），将通道文本与问题文本结合，并将其转换为模型可以处理的张量格式。调整答案跨度以匹配合并后的序列
        positive_passages = [create_reader_sample_ids(s, question) for s in positive_passages]
        negative_passages = [create_reader_sample_ids(s, question) for s in negative_passages]

        # 如果在训练集中一个样本没有正面通道，并且配置了跳过这种情况，则跳过该样本。
        if is_train_set and len(positive_passages) == 0:
            no_positive_passages += 1
            if cfg.skip_no_positves:
                continue

        # 检查 positive_passages 中是否存在一个符合特定条件（ctx.score == -1）的元素。如果存在，positives_from_gold 的值会增加 1
        # 如果没有元素符合条件（即，迭代器中没有元素），则返回 None。这里的 None 是 next 函数的第二个参数，指定当迭代器耗尽时的返回值
        # 如果找到了符合条件的元素，则返回值不会是 None，条件判断为真；如果没有找到符合条件的元素，返回值是 None，条件判断为假。
        if next(iter(ctx for ctx in positive_passages if ctx.score == -1), None):
            positives_from_gold += 1

        # 为每个样本生成一个ReaderSample对象，该对象包含问题文本、答案、正面通道和负面通道（训练集）或通道（非训练集）
        if is_train_set:
            yield ReaderSample(
                question,
                sample["answers"],
                positive_passages=positive_passages,
                negative_passages=negative_passages,
            )
        else:
            yield ReaderSample(question, sample["answers"], passages=negative_passages)

    # 记录没有正面通道的样本数量以及来自金标准的正面通道样本数量
    logger.info("no positive passages samples: %d", no_positive_passages)
    logger.info("positive passages from gold samples: %d", positives_from_gold)


def convert_retriever_results(
    is_train_set: bool,
    input_file: str,
    out_file_prefix: str,
    gold_passages_file: str,
    tensorizer: Tensorizer,
    num_workers: int = 8,
) -> List[str]:
    """
    将密集检索器（或任何兼容的文件格式）的结果转换为适合阅读器模型输入的数据，并将其序列化到一组文件中
    这一转换过程通过将输入数据分割成多个块并并行处理它们来实现，每个块的结果存储在一个单独的文件中，文件名格式为out_file_prefix.{number}.pkl
    
    is_train_set：一个布尔值，指示是否应将数据处理为训练集（即包含答案跨度检测）。
    input_file：包含要转换数据的json文件路径。
    out_file_prefix：输出文件的路径前缀。
    gold_passages_file：'金标准通道 & 问题'文件的可选路径。对于NQ（Natural Questions）数据集来说，提供此文件可以获得最佳结果。
    tensorizer：Tensorizer对象，用于将文本转换为模型输入张量。
    num_workers：用于转换的并行进程数，默认为8。
    """
    # 从input_file中加载数据，将JSON格式的数据解析为Python列表samples，每个元素代表一条样本
    with open(input_file, "r", encoding="utf-8") as f:
        samples = json.loads("".join(f.readlines()))
    logger.info("Loaded %d questions + retrieval results from %s", len(samples), input_file)
    # 根据并行工作进程的数量num_workers，将数据分割成若干个块。每个块包含的样本数大致相等，以确保并行工作的负载均衡
    workers = multiprocessing.Pool(num_workers)
    ds_size = len(samples)
    step = max(math.ceil(ds_size / num_workers), 1)
    chunks = [samples[i : i + step] for i in range(0, ds_size, step)]
    chunks = [(i, chunks[i]) for i in range(len(chunks))]

    logger.info("Split data into %d chunks", len(chunks))

    # 定义了一个局部函数_parse_batch，它使用_preprocess_reader_samples_chunk函数（一个预定义的辅助函数）来处理单个数据块。这个局部函数被配置为接受当前的输出文件前缀、金标准通道文件、Tensorizer实例和训练集标志作为参数
    processed = 0
    _parse_batch = partial(
        _preprocess_reader_samples_chunk,
        out_file_prefix=out_file_prefix,
        gold_passages_file=gold_passages_file,
        tensorizer=tensorizer,
        is_train_set=is_train_set,
    )
    # 每个数据块被独立处理并序列化到一个单独的文件中，文件名基于out_file_prefix和块的编号。处理完所有块后，函数记录并返回一个包含所有序列化文件名的列表
    serialized_files = []
    for file_name in workers.map(_parse_batch, chunks):
        processed += 1
        serialized_files.append(file_name)
        logger.info("Chunks processed %d", processed)
        logger.info("Data saved to %s", file_name)
    # 返回的是序列化结果文件的名字列表。这些文件包含了准备好的阅读器模型输入数据，适用于后续的模型训练或评估过程
    logger.info("Preprocessed data stored in %s", serialized_files)
    return serialized_files


def get_best_spans(
    tensorizer: Tensorizer,
    start_logits: List,
    end_logits: List,
    ctx_ids: List,
    max_answer_length: int,
    passage_idx: int,
    relevance_score: float,
    top_spans: int = 1,
) -> List[SpanPrediction]:
    """
    在抽取式问答模型中找到最佳的答案跨度。函数通过分析开始和结束位置的概率来识别答案，并选出得分最高的答案跨度
    tensorizer：Tensorizer对象，用于处理文本数据，包括将文本转换为模型可以处理的形式（例如，token IDs）以及将token IDs转换回文本字符串。
    start_logits和end_logits：分别代表每个token作为答案开始和结束位置的概率得分列表。
    ctx_ids：上下文token IDs列表，通常包含问题和与之相关的段落或文本。
    max_answer_length：允许的最大答案长度。
    passage_idx：当前处理的段落或文本在所有上下文中的索引。
    relevance_score：当前段落或文本的相关性评分。
    top_spans：要返回的最佳答案跨度数量，默认为1。
    """
    scores = []
    # 遍历每个可能的开始位置和结束位置（考虑到最大答案长度限制），计算每个答案跨度的得分（开始位置得分加上结束位置得分）并存储
    for (i, s) in enumerate(start_logits):
        for (j, e) in enumerate(end_logits[i : i + max_answer_length]):
            scores.append(((i, i + j), s + e))

    # 根据得分对所有可能的答案跨度进行排序，得分最高的排在前面
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    chosen_span_intervals = []
    best_spans = []

    for (start_index, end_index), score in scores:
        # 通过检查已选择的跨度列表来避免选择重叠的答案跨度
        assert start_index <= end_index
        length = end_index - start_index + 1
        assert length <= max_answer_length

        # 对于每个候选跨度，检查它是否与之前选择的任何跨度重叠，如果没有重叠，则考虑为最佳答案之一
        if any(
            [
                start_index <= prev_start_index <= prev_end_index <= end_index
                or prev_start_index <= start_index <= end_index <= prev_end_index
                for (prev_start_index, prev_end_index) in chosen_span_intervals
            ]
        ):
            continue

        # extend bpe subtokens to full tokens
        # 使用_extend_span_to_full_words函数扩展BPE（Byte Pair Encoding）子tokens到完整的单词，确保答案是完整的单词而非片段
        start_index, end_index = _extend_span_to_full_words(tensorizer, ctx_ids, (start_index, end_index))

        # 对于每个选定的最佳答案跨度，将跨度内的token IDs转换回文本字符串作为预测的答案
        predicted_answer = tensorizer.to_string(ctx_ids[start_index : end_index + 1])
        # 创建SpanPrediction对象，包含预测的答案文本、得分、相关性评分、段落索引和上下文IDs
        best_spans.append(SpanPrediction(predicted_answer, score, relevance_score, passage_idx, ctx_ids))
        chosen_span_intervals.append((start_index, end_index))

        # 如果已找到指定数量的最佳答案跨度（top_spans），则停止搜索
        if len(chosen_span_intervals) == top_spans:
            break
    # 返回包含最佳答案跨度的列表，每个答案跨度都包含答案文本、得分等信息
    return best_spans


def _select_reader_passages(
    sample: Dict,
    question: str,
    tensorizer: Tensorizer ,
    gold_passage_map: Optional[Dict[str, ReaderPassage]],
    gold_page_only_positives: bool,
    max_positives: int,
    max1_negatives: int,
    max2_negatives: int,
    max_retriever_passages: int,
    include_gold_passage: bool,
    is_train_set: bool,
) -> Tuple[List[ReaderPassage], List[ReaderPassage]]:
    """ 
    根据问题和答案的相关性，从一组候选文段中筛选出有助于模型学习的正面（包含答案的）和负面（不包含答案的）样本
    
    sample: 包含候选文段(ctxs)和答案(answers)的字典。
    question: 当前的问题文本。
    tensorizer: 用于将文本转换为张量的工具，方便后续模型处理。
    gold_passage_map: 一个字典，键为问题文本，值为含有正确答案的“金标准”文段。
    gold_page_only_positives: 布尔值，指示是否只从“金标准”页面选取正面样本。
    max_positives, max1_negatives, max2_negatives: 分别控制正面样本、负面样本的最大数量。
    max_retriever_passages: 考虑的最大候选文段数量。
    include_gold_passage: 布尔值，指示是否包含“金标准”文段作为正面样本。
    is_train_set: 布尔值，指示当前是否处于训练阶段。
    """
    answers = sample["answers"]

    # 将候选文段列表(ctxs)限制在max_retriever_passages指定的数量内，并尝试为每个答案生成对应的张量
    ctxs = [ReaderPassage(**ctx) for ctx in sample["ctxs"]][0:max_retriever_passages]
    answers_token_ids = [tensorizer.text_to_tensor(a, add_special_tokens=False) for a in answers]

    # 根据是否为训练集(is_train_set)，区分选择正面和负面样本的策略
    if is_train_set:
        positive_samples = list(filter(lambda ctx: ctx.has_answer, ctxs))
        negative_samples = list(filter(lambda ctx: not ctx.has_answer, ctxs))
    else:
        positive_samples = []
        negative_samples = ctxs

    # 如果设置了gold_page_only_positives并提供了gold_passage_map，则尝试从与“金标准”页面相关的文段中筛选正面样本
    positive_ctxs_from_gold_page = (
        list(
            filter(
                lambda ctx: _is_from_gold_wiki_page(gold_passage_map, ctx.title, question),
                positive_samples,
            )
        )
        if gold_page_only_positives and gold_passage_map
        else []
    )

    def find_answer_spans(ctx: ReaderPassage):
        """ 
        在文本段落中找到答案的位置
        
        ReaderPassage可能是一个自定义的类，用于存储与阅读理解相关的数据，如段落文本、答案和其他相关信息
        """
        if ctx.has_answer: # 存在答案
            if ctx.passage_token_ids is None:
                # 如果passage_token_ids为空，则将段落文本ctx.passage_text转换为张量形式，存储回ctx.passage_token_ids。这个过程不添加特殊的标记
                ctx.passage_token_ids = tensorizer.text_to_tensor(ctx.passage_text, add_special_tokens=False)

            # 遍历所有答案，调用_find_answer_positions函数来找出每个答案在文本中的位置。这里假设answers是一个包含答案的列表，而answers_token_ids是这些答案对应的分词后的张量形式
            answer_spans = [
                _find_answer_positions(ctx.passage_token_ids, answers_token_ids[i]) for i in range(len(answers))
            ]

            # 将找到的答案位置列表扁平化。由于_find_answer_positions可能返回一个列表的列表，这一行代码将所有这些列表合并成一个单一的列表
            answer_spans = [item for sublist in answer_spans for item in sublist]
            # 使用filter函数移除所有空值。这步确保只有有效的答案位置被保留
            answers_spans = list(filter(None, answer_spans))
            # 将处理后的答案位置列表存储回ctx对象的answers_spans属性中
            ctx.answers_spans = answers_spans

            # 日志记录的警告，如果没有找到答案，就记录一条警告信息
            if not answers_spans:
                logger.warning(
                    "No answer found in passage id=%s text=%s, answers=%s, question=%s",
                    ctx.id,
                    "",  # ctx.passage_text
                    answers,
                    question,
                )
            # 更新ctx对象的has_answer属性，根据是否找到答案位置来设置为真或假
            ctx.has_answer = bool(answers_spans)
        return ctx

    # 通过find_answer_spans函数检查positive_ctxs_from_gold_page（从“金标准”页面筛选的正面样本）中的每个文段，确定它们是否真的包含了答案。这一步是通过搜索答案在文段中的位置来实现的。如果文段确实包含答案，那么这个文段就被保留下来
    selected_positive_ctxs = list(
        filter(
            lambda ctx: ctx.has_answer,
            [find_answer_spans(ctx) for ctx in positive_ctxs_from_gold_page],
        )
    )

    # 如果在“金标准”页面选定的正面样本中没有找到任何含答案的文段，那么代码会回退到positive_samples（所有正面样本）中去寻找，并限制选取的最大数量为max_positives
    if not selected_positive_ctxs:  # fallback to positive ctx not from gold pages
        selected_positive_ctxs = list(
            filter(
                lambda ctx: ctx.has_answer,
                [find_answer_spans(ctx) for ctx in positive_samples],
            )
        )[0:max_positives]

    # 如果设置了包含“金标准”文段（include_gold_passage为True），并且这个“金标准”文段还没有被包含在当前选定的正面样本中，那么这个文段会被添加到正面样本列表中。这一步骤包括了检查该“金标准”文段是否真的包含答案，并在确认包含答案的情况下，将其添加为正面样本
    if include_gold_passage and question in gold_passage_map:
        gold_passage = gold_passage_map[question]
        included_gold_passage = next(
            iter(ctx for ctx in selected_positive_ctxs if ctx.passage_text == gold_passage.passage_text),
            None,
        )
        if not included_gold_passage:
            gold_passage.has_answer = True
            gold_passage = find_answer_spans(gold_passage)
            if not gold_passage.has_answer:
                # 如果在“金标准”文段中没有找到答案，将通过日志记录一个警告
                logger.warning("No answer found in gold passage: %s", gold_passage)
            else:
                selected_positive_ctxs.append(gold_passage)

    # 计算负面样本的数量，其逻辑依赖于是否处于训练阶段。如果是训练集，那么负面样本的数量将基于选定正面样本的数量动态计算，但不超过max1_negatives和max2_negatives设定的上限。如果不是训练集，那么使用DEFAULT_EVAL_PASSAGES作为负面样本的数量
    max_negatives = (
        min(max(10 * len(selected_positive_ctxs), max1_negatives), max2_negatives)
        if is_train_set
        else DEFAULT_EVAL_PASSAGES
    )
    # 从negative_samples（所有负面样本）中选取计算得到的数量上限的样本作为最终的负面样本列表
    negative_samples = negative_samples[0:max_negatives]
    # 返回两个列表：selected_positive_ctxs（选定的正面样本）和negative_samples（选定的负面样本）
    return selected_positive_ctxs, negative_samples


def _find_answer_positions(ctx_ids: T, answer: T) -> List[Tuple[int, int]]:
    """ 
    找出一个序列（答案）在另一个更长序列（上下文）中所有出现的位置
    """
    c_len = ctx_ids.size(0) # 获取序列的长度
    a_len = answer.size(0)
    answer_occurences = [] # 存储找到的答案的所有出现位置
    for i in range(0, c_len - a_len + 1):
        # 检查从i开始、长度为a_len的ctx_ids的子序列是否与answer完全相同。如果完全相同，.all()方法会返回True
        if (answer == ctx_ids[i : i + a_len]).all():
            # 将匹配项的起始和结束索引作为一个元组添加到answer_occurences列表中
            answer_occurences.append((i, i + a_len - 1))
    return answer_occurences


def _concat_pair(t1: T, t2: T, middle_sep: T = None, tailing_sep: T = None):
    """ 
    将两个张量（Tensor）拼接在一起，并可选地在它们之间以及后面添加分隔符
    
    t1 和 t2 是要被拼接的两个张量。
    middle_sep 是可选参数，默认值为 None，指定在 t1 和 t2 之间插入的分隔符。
    tailing_sep 也是可选参数，默认值为 None，指定在 t2 后面插入的分隔符
    """
    middle = [middle_sep] if middle_sep else []
    # 根据输入的参数构建一个包含所有元素（包括可能的分隔符）的列表
    r = [t1] + middle + [t2] + ([tailing_sep] if tailing_sep else [])
    # 返回值是一个元组，第一个元素是拼接后的张量，第二个元素是 t1 在第一个维度上的大小加上 middle 列表的长度（如果有中间分隔符，长度为1；否则为0）。这个返回值可能用于指示拼接操作中原始张量 t1 的贡献或位置
    return torch.cat(r, dim=0), t1.size(0) + len(middle)


def _get_gold_ctx_dict(file: str) -> Tuple[Dict[str, ReaderPassage], Dict[str, str]]:
    """ 
    从给定的文件中读取“gold”上下文数据，并将其组织成两个字典：一个用于映射问题（及其令牌化形式）到对应的阅读器通道 (ReaderPassage)，另一个用于映射令牌化形式的问题回原始问题
    """
    gold_passage_infos = {}  # question|question_tokens -> ReaderPassage (with title and gold ctx)
    original_questions = {}  # question from tokens -> original question (NQ only)

    # 以只读模式打开指定的文件，然后使用 json.load 读取文件内容。假定文件内容是JSON格式，并且期望数据在文件的 "data" 键下
    with open(file, "r", encoding="utf-8") as f:
        logger.info("Reading file %s" % file)
        data = json.load(f)["data"]

    for sample in data:
        # 提取问题（question）和令牌化形式的问题（question_from_tokens），如果样本中没有提供令牌化形式，则使用问题本身作为令牌化形式
        question = sample["question"]
        question_from_tokens = sample["question_tokens"] if "question_tokens" in sample else question
        # 更新 original_questions 字典，将令牌化形式的问题映射回原始问题
        original_questions[question_from_tokens] = question
        # 提取样本的标题（title）和上下文（context），创建一个 ReaderPassage 对象（假定有一个 example_id 字段作为唯一标识符）
        title = sample["title"].lower()
        context = sample["context"]  # Note: This one is cased
        rp = ReaderPassage(sample["example_id"], text=context, title=title)
        # 如果问题（或其令牌化形式）已经在 gold_passage_infos 字典中，则记录关于重复问题的信息（包括新旧标题和上下文）
        if question in gold_passage_infos:
            logger.info("Duplicate question %s", question)
            rp_exist = gold_passage_infos[question]
            logger.info(
                "Duplicate question gold info: title new =%s | old title=%s",
                title,
                rp_exist.title,
            )
            logger.info("Duplicate question gold info: new ctx =%s ", context)
            logger.info("Duplicate question gold info: old ctx =%s ", rp_exist.passage_text)
        # 将问题（及其令牌化形式）映射到 ReaderPassage 对象，并更新 gold_passage_infos 字典
        gold_passage_infos[question] = rp
        gold_passage_infos[question_from_tokens] = rp
    return gold_passage_infos, original_questions


def _is_from_gold_wiki_page(gold_passage_map: Dict[str, ReaderPassage], passage_title: str, question: str):
    """ 
    判断给定的文章标题是否来自于与特定问题相关的“黄金”维基页面
    “黄金”维基页面指的是对于特定问题，被认为是权威或特别相关的维基页面
    
    gold_passage_map: 一个字典，键为字符串类型，值为 ReaderPassage 类型。这个字典映射了问题到其对应的“黄金”维基页面信息。
    passage_title: 一个字符串，代表待检查的文章标题。
    question: 一个字符串，代表查询的问题。
    """
    # 从 gold_passage_map 中获取与 question 相关的“黄金”页面信息。如果找到了对应的信息，gold_info 将被赋值为该信息；否则，gold_info 将被赋值为 None
    gold_info = gold_passage_map.get(question, None)
    if gold_info:
        # 如果它们相等，意味着给定的文章标题确实来自于与问题相关的“黄金”维基页面，函数返回 True
        return passage_title.lower() == gold_info.title.lower()
    return False


def _extend_span_to_full_words(tensorizer: Tensorizer, tokens: List[int], span: Tuple[int, int]) -> Tuple[int, int]:
    """ 
    在给定的令牌序列中扩展一个词组（span）的范围，以确保该范围包含完整的词而不是被截断的子词
    tensorizer: Tensorizer：一个对象，其具有一个方法 is_sub_word_id，该方法能够判断给定的令牌ID是否为子词令牌。
    tokens: List[int]：一个整数列表，表示令牌ID序列。这些令牌ID对应于一段文本的令牌化表示。
    span: Tuple[int, int]：一个元组，表示要扩展的词组的起始和结束索引。
    -> Tuple[int, int]：函数返回值的类型注释，表示返回一个包含两个整数的元组。
    """
    start_index, end_index = span # 获取起始和结束索引
    max_len = len(tokens) # 获取令牌序列的长度
    while start_index > 0 and tensorizer.is_sub_word_id(tokens[start_index]):
        # 检查当前 start_index 是否指向一个子词令牌。如果是，且 start_index 大于0（确保不越界），则将 start_index 向前移动一个位置。这个循环的目的是将起始索引移动到当前词组的第一个完整词的开始位置
        start_index -= 1

    while end_index < max_len - 1 and tensorizer.is_sub_word_id(tokens[end_index + 1]):
        # 检查 end_index + 1 是否指向一个子词令牌。如果是，且 end_index 小于 max_len - 1（确保不越界），则将 end_index 向后移动一个位置。这个循环的目的是将结束索引移动到当前词组的最后一个完整词的结束位置。
        end_index += 1

    return start_index, end_index


def _preprocess_reader_samples_chunk(
    samples: List,
    out_file_prefix: str,
    gold_passages_file: str,
    tensorizer: Tensorizer,
    is_train_set: bool,
) -> str:
    """ 
    处理和序列化一批问答样本数据
    
    samples: 包含样本的列表，其中第一个元素预期是批处理ID（chunk_id），第二个元素是样本数据。
    out_file_prefix: 输出文件的前缀，用于构造最终的文件名。
    gold_passages_file: 包含“gold”通道数据的文件路径，用于预处理。
    tensorizer: 一个 Tensorizer 对象，用于样本的张量化处理。
    is_train_set: 一个布尔值，指示当前处理的数据集是否为训练集。
    函数返回一个字符串，即序列化数据存储的文件名。
    """
    chunk_id, samples = samples # 提取批处理ID (chunk_id) 和样本数据
    logger.info("Start batch %d", len(samples)) # 记录开始处理当前批次的信息，包括样本的数量
    # 进行样本预处理，返回一个迭代器
    iterator = preprocess_retriever_data(
        samples,
        gold_passages_file,
        tensorizer,
        is_train_set=is_train_set,
    )

    results = [] # 存储处理后的样本

    # 使用 tqdm 包装 iterator 来显示处理进度条。遍历迭代器中的每个结果 r，调用其 on_serialize 方法进行序列化准备，然后将处理后的结果添加到 results 列表中
    iterator = tqdm(iterator)
    for i, r in enumerate(iterator):
        r.on_serialize()
        results.append(r)

    # 构造输出文件名 out_file，使用 out_file_prefix、chunk_id 和文件扩展名 .pkl。然后以二进制写入模式打开文件，记录序列化操作的信息，并使用 pickle.dump 将 results 序列化到文件中
    out_file = out_file_prefix + "." + str(chunk_id) + ".pkl"
    with open(out_file, mode="wb") as f:
        logger.info("Serialize %d results to %s", len(results), out_file)
        pickle.dump(results, f)
    return out_file # 返回包含序列化数据的输出文件名
