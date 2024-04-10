#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The reader model code + its utilities (loss computation and input batch tensor generator)
"""

import collections
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor as T
from torch.nn import CrossEntropyLoss

from dpr.data.reader_data import ReaderSample, ReaderPassage
from dpr.utils.model_utils import init_weights
logger = logging.getLogger()

ReaderBatch = collections.namedtuple(
    "ReaderBatch", ["input_ids", "start_positions", "end_positions", "answers_mask", "token_type_ids"]
)


class Reader(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size):
        super(Reader, self).__init__()
        self.encoder = encoder
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.qa_classifier = nn.Linear(hidden_size, 1)
        init_weights([self.qa_outputs, self.qa_classifier])

    def forward(
        self,
        input_ids: T,
        attention_mask: T,
        toke_type_ids: T,
        start_positions=None,
        end_positions=None,
        answer_mask=None,
    ):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        N, M, L = input_ids.size()
        start_logits, end_logits, relevance_logits = self._forward(
            input_ids.view(N * M, L),
            attention_mask.view(N * M, L),
            toke_type_ids.view(N * M, L),
        )
        if self.training:
            return compute_loss(
                start_positions, end_positions, answer_mask, start_logits, end_logits, relevance_logits, N, M
            )

        return start_logits.view(N, M, L), end_logits.view(N, M, L), relevance_logits.view(N, M)

    def _forward(self, input_ids, attention_mask, toke_type_ids: T):
        sequence_output, _pooled_output, _hidden_states = self.encoder(input_ids, toke_type_ids, attention_mask)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        rank_logits = self.qa_classifier(sequence_output[:, 0, :])
        return start_logits, end_logits, rank_logits


def compute_loss(start_positions, end_positions, answer_mask, start_logits, end_logits, relevance_logits, N, M):
    """ 
    在训练阅读理解模型时计算损失
    
    start_positions, end_positions：每个答案的开始和结束位置的索引。
    answer_mask：一个掩码，标示每个答案位置是否有效。
    start_logits, end_logits：模型预测的每个位置作为答案开始和结束的逻辑回归值。
    relevance_logits：模型预测的段落与问题相关性的逻辑回归值。
    N：批次中样本的数量。
    M：每个样本中考虑的段落数量。
    """
    # 将位置索引、逻辑回归值和掩码调整为(N*M, -1)的形状，以便于后续的批处理操作。
    start_positions = start_positions.view(N * M, -1)
    end_positions = end_positions.view(N * M, -1)
    answer_mask = answer_mask.view(N * M, -1)

    start_logits = start_logits.view(N * M, -1)
    end_logits = end_logits.view(N * M, -1)
    # 将relevance_logits调整为(N, M)的形状，表示每个样本中每个段落的相关性逻辑回归值
    relevance_logits = relevance_logits.view(N * M)

    # 类型转换操作，将answer_mask张量的数据类型转换为torch.FloatTensor，即32位浮点数
    answer_mask = answer_mask.type(torch.FloatTensor).cuda()

    # 原地（in-place）操作，用于将张量中的元素限制在指定的范围内。如果张量中的元素小于min，则会被设置为min；如果大于max，则会被设置为max
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)
    # reduce参数用来指定是否应用降维操作。当reduce=False时，损失函数不会返回一个标量损失值，而是返回每个样本的损失值
    # ignored_index = start_logits.size(1)意味着忽略最后一个索引，这通常是为了避免计算超出序列长度的位置上的损失
    loss_fct = CrossEntropyLoss(reduce=False, ignore_index=ignored_index)

    # compute switch loss
    relevance_logits = relevance_logits.view(N, M)
    # 创建了一个形状为(N,)的全0张量，表示每个样本中选择第一个段落作为最相关段落的标签（假定最相关的段落标签为0）
    switch_labels = torch.zeros(N, dtype=torch.long).cuda()
    # 使用CrossEntropyLoss损失函数计算切换损失。这里，relevance_logits中的每一行对应于一个样本中所有段落的相关性得分，switch_labels是全零的，表示最相关的段落是第一个。通过这种方式，模型被训练以将最高的相关性得分分配给第一个段落
    switch_loss = torch.sum(loss_fct(relevance_logits, switch_labels))

    # compute span loss
    # 使用CrossEntropyLoss计算每个答案开始位置和结束位置的损失。这一步包括迭代每一列的start_positions和end_positions（通过torch.unbind分解），以及相应的answer_mask，计算每个可能答案位置的损失并乘以相应的_span_mask来忽略那些标记为无效的答案位置
    start_losses = [
        (loss_fct(start_logits, _start_positions) * _span_mask)
        for (_start_positions, _span_mask) in zip(
            torch.unbind(start_positions, dim=1), torch.unbind(answer_mask, dim=1)
        )
    ]

    end_losses = [
        (loss_fct(end_logits, _end_positions) * _span_mask)
        for (_end_positions, _span_mask) in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_mask, dim=1))
    ]
    # 通过将所有开始位置的损失和所有结束位置的损失相加，然后通过torch.cat将它们连接起来，形成一个损失张量loss_tensor。这个张量随后被调整形状为(N, M, -1)，并通过.max(dim=1)[0]选择每个样本中损失最大的那个段落，这代表每个问题的最终损失
    loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + torch.cat(
        [t.unsqueeze(1) for t in end_losses], dim=1
    )

    loss_tensor = loss_tensor.view(N, M, -1).max(dim=1)[0]
    # 调用自定义函数_calc_mml（可能是计算最大边缘似然的函数）来处理loss_tensor，得到最终的跨度损失span_loss
    span_loss = _calc_mml(loss_tensor)
    return span_loss + switch_loss


def create_reader_input(
    pad_token_id: int,
    samples: List[ReaderSample],
    passages_per_question: int,
    max_length: int,
    max_n_answers: int,
    is_train: bool,
    shuffle: bool,
    sep_token_id: int,
) -> ReaderBatch:
    """
    将一组阅读理解任务中的样本（ReaderSample）转换为适用于模型的批处理数据格式（ReaderBatch）
    
    pad_token_id：填充token的ID，用于将序列填充到指定的最大长度。
    samples：待处理的样本列表，每个样本包括问题及其正面和负面上下文（或段落）。
    passages_per_question：每个问题要考虑的段落数量。
    max_length：模型输入序列的最大长度。
    max_n_answers：每个问题的最大答案数量。
    is_train：标记是否为训练集样本。
    shuffle：是否随机选择段落。
    sep_token_id：分隔token的ID，用于区分不同的序列或段落。
    """
    input_ids = []
    start_positions = []
    end_positions = []
    answers_masks = []
    token_type_ids = []
    # 创建一个填充序列empty_sequence，长度为max_length，全部填充pad_token_id。
    empty_sequence = torch.Tensor().new_full((max_length,), pad_token_id, dtype=torch.long)

    for sample in samples:
        # 遍历每个sample，根据是否为训练集来选择使用正面上下文（positive_ctxs）还是负面上下文（negative_ctxs）
        positive_ctxs = sample.positive_passages
        negative_ctxs = sample.negative_passages if is_train else sample.passages

        # 调用_create_question_passages_tensors函数处理每个样本，生成输入ID、答案开始位置、答案结束位置、答案掩码和token类型ID。
        sample_tensors = _create_question_passages_tensors(
            positive_ctxs,
            negative_ctxs,
            passages_per_question,
            empty_sequence,
            max_n_answers,
            pad_token_id,
            sep_token_id,
            is_train,
            is_random=shuffle,
        )
        if not sample_tensors:
            # 如果_create_question_passages_tensors返回结果为空（可能因为没有有效的段落组合），则跳过当前样本
            logger.debug("No valid passages combination for question=%s ", sample.question)
            continue
        sample_input_ids, starts_tensor, ends_tensor, answer_mask, sample_ttids = sample_tensors
        input_ids.append(sample_input_ids)
        token_type_ids.append(sample_ttids)
        if is_train:
            start_positions.append(starts_tensor)
            end_positions.append(ends_tensor)
            answers_masks.append(answer_mask)
    # 使用torch.cat和torch.stack将列表中的数据整理成批处理所需的张量格式
    input_ids = torch.cat([ids.unsqueeze(0) for ids in input_ids], dim=0)
    token_type_ids = torch.cat([ids.unsqueeze(0) for ids in token_type_ids], dim=0)  # .unsqueeze(0)

    if is_train:
        # 对于训练集样本，会包括答案的开始和结束位置以及答案掩码；对于非训练集样本，这些数据可能不包括。
        start_positions = torch.stack(start_positions, dim=0)
        end_positions = torch.stack(end_positions, dim=0)
        answers_masks = torch.stack(answers_masks, dim=0)

    # 返回一个ReaderBatch实例，包含了整个批处理的input_ids、start_positions、end_positions、answers_masks和token_type_ids。
    return ReaderBatch(input_ids, start_positions, end_positions, answers_masks, token_type_ids)


def _calc_mml(loss_tensor):
    # 计算给定损失张量（loss_tensor）的边缘似然（Marginal Maximum Likelihood, MML）
    """ 
    (loss_tensor == 0).float()会生成一个与loss_tensor形状相同的布尔张量，其中loss_tensor中的每个零值都会被标记为1.0（True），非零值标记为0.0（False）。
    接着，-1e10 * (loss_tensor == 0).float()将这个布尔张量乘以-1e10（一个非常大的负数），这样原始张量中为零的元素在经过指数运算exp时几乎贡献为零，避免了直接对零取对数造成的数学错误或不稳定。
    torch.exp(-loss_tensor - 1e10 * (loss_tensor == 0).float())计算损失张量的负值的指数，这一步是为了将损失转换为概率（或似然）的形式。
    torch.sum(..., 1)对这个经过转换的张量在第二个维度（即索引为1的维度，对应每个样本）上求和，以计算每个样本的边缘似然
    """
    marginal_likelihood = torch.sum(torch.exp(-loss_tensor - 1e10 * (loss_tensor == 0).float()), 1)
    """ 
    计算边缘似然的对数。为了避免对零取对数导致的数值不稳定问题，对边缘似然等于零的情况进行了特殊处理：当marginal_likelihood为零时，通过(marginal_likelihood == 0).float()产生的张量在对应位置为1.0，与torch.ones(loss_tensor.size(0)).cuda()相乘（后者生成一个所有元素为1的张量，并将其放到CUDA内存上以便在GPU上进行计算），最终实现了当边缘似然为零时，在对数计算前给它加上一个很小的值（1.0）以避免数学错误。
    最后，通过-torch.sum(...)对所有样本的调整后的边缘似然取对数求和，并取负号。这是因为在许多优化问题中，我们最大化似然的对数，这在数学上等价于最小化负的似然的对数
    """
    return -torch.sum(
        torch.log(marginal_likelihood + torch.ones(loss_tensor.size(0)).cuda() * (marginal_likelihood == 0).float())
    )


def _pad_to_len(seq: T, pad_id: int, max_len: int):
    # 对序列进行填充或截断以达到指定的最大长度 max_len
    s_len = seq.size(0)
    if s_len > max_len:
        # 如果输入序列的长度 s_len 超过了允许的最大长度 max_len，则通过 seq[0:max_len] 截断序列，保留从开始到 max_len 的部分
        return seq[0:max_len]
    """
    torch.Tensor().new_full((max_len - s_len,), pad_id, dtype=torch.long) 创建一个新的张量，其中所有值都设为 pad_id，大小为 (max_len - s_len,)，即缺少的部分长度。
    torch.cat([seq, ...], dim=0) 将原始序列 seq 和填充张量合并在一起，从而得到一个长度为 max_len 的新序列 
    """
    return torch.cat([seq, torch.Tensor().new_full((max_len - s_len,), pad_id, dtype=torch.long)], dim=0)


def _get_answer_spans(idx, positives: List[ReaderPassage], max_len: int):
    """ 
    从指定正面样本中提取答案范围，并确保这些答案范围都在给定的最大长度内
    idx: 指定从哪个正面样本中提取答案范围。这是一个索引值，指向positives列表中的特定元素。
    positives: ReaderPassage对象的列表，每个对象都代表一个正面样本，即包含答案的文段。
    max_len: 答案范围必须在此长度之内。这个参数用于确保提取的答案不会超出处理的最大长度限制
    """
    positive_a_spans = positives[idx].answers_spans # 获取指定正面样本中所有答案的范围。answers_spans是一个列表，其中的每个元素都是一个元组(start, end)，表示答案在文段中的起始和结束位置
    # 使用列表推导式来筛选出那些既不超过文段起始位置的最大长度max_len，也不超过文段结束位置的最大长度max_len的答案范围。这确保了提取的答案范围都在可处理的长度范围内
    return [span for span in positive_a_spans if (span[0] < max_len and span[1] < max_len)]


def _get_positive_idx(positives: List[ReaderPassage], max_len: int, is_random: bool):
    """ 
    从一组正面实例中选择一个具体的实例索引
    正面实例（positives）通常指与查询或问题相关联且包含正确答案的数据点
    
    positives是ReaderPassage对象的列表，max_len是一个整数，表示最大长度限制，is_random是一个布尔值，指示是否随机选择正面实例。
    """
    # 如果is_random为真，则随机选择一个索引；否则，默认选择列表中的第一个实例（索引为0）
    positive_idx = np.random.choice(len(positives)) if is_random else 0

    # 检查通过选定的索引positive_idx得到的实例是否包含至少一个有效的答案跨度
    if not _get_answer_spans(positive_idx, positives, max_len):
        # 如果选定的正面实例不包含有效的答案跨度，这行代码会遍历正面实例的列表，查找第一个包含至少一个有效答案跨度的实例，并更新positive_idx为该实例的索引。如果所有实例都不符合条件，positive_idx将被设置为None
        positive_idx = next((i for i in range(len(positives)) if _get_answer_spans(i, positives, max_len)), None)
    return positive_idx # 返回选定的正面实例索引


def _create_question_passages_tensors(
    positives: List[ReaderPassage],
    negatives: List[ReaderPassage],
    total_size: int,
    empty_ids: T,
    max_n_answers: int,
    pad_token_id: int,
    sep_token_id: int,
    is_train: bool,
    is_random: bool = True,
    first_segment_ttid: int = 0,
):
    """ 
    处理阅读理解任务中问题和段落的数据，并将其转换为模型可以处理的格式
    
    正面实例（positives）、负面实例（negatives）、总尺寸（total_size）、空ID（empty_ids）、最大答案数（max_n_answers）、填充符ID（pad_token_id）、分隔符ID（sep_token_id）、训练标志（is_train）、随机标志（is_random），以及第一个片段的token类型ID（first_segment_ttid）
    """
    max_len = empty_ids.size(0) # 指定最大序列长度
    if is_train:
        # 如果处于训练模式，函数会选择一个正面实例，并获取该实例的答案跨度。使用_get_positive_idx函数基于是否随机选择正面实例
        positive_idx = _get_positive_idx(positives, max_len, is_random)
        if positive_idx is None:
            return None

        # 获取选中正面实例的答案跨度positive_a_spans，限制到max_n_answers
        positive_a_spans = _get_answer_spans(positive_idx, positives, max_len)[0:max_n_answers]

        # 从答案跨度中提取答案的开始和结束位置，以及确保这些位置在最大长度范围内
        answer_starts = [span[0] for span in positive_a_spans]
        answer_ends = [span[1] for span in positive_a_spans]

        assert all(s < max_len for s in answer_starts)
        assert all(e < max_len for e in answer_ends)

        # 对选中的正面实例序列ID进行填充，以满足最大长度要求
        positive_input_ids = _pad_to_len(positives[positive_idx].sequence_ids, pad_token_id, max_len)

        # 创建两个全0的长整型张量answer_starts_tensor和answer_ends_tensor，其形状由total_size（数据总量）和max_n_answers（最大答案数量）确定。这些张量将用于存储答案的开始和结束位置
        # 使用torch.tensor(answer_starts)和torch.tensor(answer_ends)填充这些张量的第一个元素（对应于选中的正面实例）的相应答案开始和结束位置
        answer_starts_tensor = torch.zeros((total_size, max_n_answers)).long()
        answer_starts_tensor[0, 0 : len(answer_starts)] = torch.tensor(answer_starts)

        answer_ends_tensor = torch.zeros((total_size, max_n_answers)).long()
        answer_ends_tensor[0, 0 : len(answer_ends)] = torch.tensor(answer_ends)

        # answer_mask是一个形状与答案开始和结束位置张量相同的长整型张量，用于标识有效答案位置。通过将有效答案位置设为1，其余为0，来指示哪些位置包含有效答案
        answer_mask = torch.zeros((total_size, max_n_answers), dtype=torch.long)
        answer_mask[0, 0 : len(answer_starts)] = torch.tensor([1 for _ in range(len(answer_starts))])

        positives_selected = [positive_input_ids]

    else:
        # 在非训练模式（else部分），不进行特殊处理，直接设置空集合或None。
        positives_selected = []
        answer_starts_tensor = None
        answer_ends_tensor = None
        answer_mask = None

    # 根据is_random标志随机或顺序选择负面实例的索引（negative_idxs），并确保选择的负面实例数量加上正面实例数量不超过total_size指定的总数量。
    positives_num = len(positives_selected)
    negative_idxs = np.random.permutation(range(len(negatives))) if is_random else range(len(negatives) - positives_num)

    negative_idxs = negative_idxs[: total_size - positives_num]

    # 对选择的负面实例进行序列填充，与正面实例一起形成negatives_selected列表
    negatives_selected = [_pad_to_len(negatives[i].sequence_ids, pad_token_id, max_len) for i in negative_idxs]
    negatives_num = len(negatives_selected)

    # input_ids通过将正面和负面实例的序列ID堆叠起来生成
    input_ids = torch.stack([t for t in positives_selected + negatives_selected], dim=0)

    # toke_type_ids通过_create_token_type_ids函数根据input_ids、sep_token_id（分隔符ID）和first_segment_ttid（第一个片段的token类型ID）生成，用于模型区分不同片段
    toke_type_ids = _create_token_type_ids(input_ids, sep_token_id, first_segment_ttid)

    # 如果正面和负面实例的总数少于total_size，使用empty_ids填充剩余空间。这是通过克隆empty_ids并调整其形状完成的
    if positives_num + negatives_num < total_size:
        empty_negatives = [empty_ids.clone().view(1, -1) for _ in range(total_size - (positives_num + negatives_num))]
        empty_token_type_ids = [
            empty_ids.clone().view(1, -1) for _ in range(total_size - (positives_num + negatives_num))
        ]

        input_ids = torch.cat([input_ids, *empty_negatives], dim=0)
        toke_type_ids = torch.cat([toke_type_ids, *empty_token_type_ids], dim=0)

    # 返回input_ids（输入序列ID）、answer_starts_tensor（答案开始位置）、answer_ends_tensor（答案结束位置）、answer_mask（答案掩码）以及toke_type_ids（token类型ID）
    return input_ids, answer_starts_tensor, answer_ends_tensor, answer_mask, toke_type_ids


def _create_token_type_ids(input_ids: torch.Tensor, sep_token_id: int, first_segment_ttid: int = 0):
    """ 
    为输入的标识符序列创建令牌类型标识符（Token Type IDs）
    input_ids: 一个torch.Tensor对象，包含了编码后的标识符序列，其中可能包含一个或多个分隔符标识符（[SEP]）用于区分不同的序列。
    sep_token_id: 分隔符标识符的ID。在BERT等模型中，[SEP]标识符用于分隔双序列（如问题和答案）。
    first_segment_ttid: 第一序列的令牌类型标识符（Token Type ID）。默认值为0，通常用于指示序列的第一部分。第二部分的ID通常是1（除非第一部分的ID被设置为1，那么第二部分的ID就会被设置为0）。
    """
    # 使用torch.full创建一个与input_ids形状相同、但全部填充为0的新torch.Tensor对象token_type_ids
    token_type_ids = torch.full(input_ids.shape, fill_value=0)
    # 使用torch.nonzero找出input_ids中所有分隔符标识符sep_token_id的索引位置
    sep_tokens_indexes = torch.nonzero(input_ids == sep_token_id)
    bsz = input_ids.size(0) # 计算输入序列的批次大小（bsz）
    # 根据first_segment_ttid的值确定第二序列的令牌类型标识符（second_ttid）
    second_ttid = 0 if first_segment_ttid == 1 else 1

    # 遍历每个序列（批次中的每个样本），并使用分隔符索引来更新token_type_ids。对于每个样本，序列中第一个[SEP]标识符及其之前的部分被标记为first_segment_ttid，而第一个[SEP]标识符之后的部分被标记为second_ttid
    for i in range(bsz):
        token_type_ids[i, 0 : sep_tokens_indexes[2 * i, 1] + 1] = first_segment_ttid
        token_type_ids[i, sep_tokens_indexes[2 * i, 1] + 1 :] = second_ttid
    return token_type_ids
