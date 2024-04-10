#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
import logging
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import BiEncoderSample
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState

logger = logging.getLogger(__name__)

BiEncoderBatch = collections.namedtuple(
    # "BiEncoderInput" 是定义的 namedtuple 类的名称，这是当你创建该 namedtuple 实例时所使用的类名
    #  Python 解释器只关心变量名 BiEncoderBatch 指向的 namedtuple 类型，而 "BiEncoderInput" 实际上只是在实例化 namedtuple 时使用的类型名称
    "BiENcoderInput",
    [
        "question_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
        "encoder_type",
    ],
)
# TODO: it is only used by _select_span_with_token. Move them to utils
# 创建一个随机数生成器，种子值为0
rnd = random.Random(0)


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    计算两组向量之间的点积得分, 评估查询向量与一组上下文向量的相似度
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    # 使用torch.transpose(ctx_vectors, 0, 1)将ctx_vectors的维度进行转置，以便它们的内部维度（即每个向量的维度D）对齐。然后，使用torch.matmul计算两个张量的矩阵乘法，得到的结果是查询向量和上下文向量之间的点积得分
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""
    # BiEncoder 类是一个双编码器模型的实现，用于处理查询/问题（question）和上下文/段落（context）的编码

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        """ 
        question_model: 用于编码问题的模型。
        ctx_model: 用于编码上下文的模型。
        fix_q_encoder: 布尔值，用于决定是否固定问题编码器的参数。
        fix_ctx_encoder: 布尔值，用于决定是否固定上下文编码器的参数
        """
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
    ) -> (T, T, T):
        """ 
        sub_model: 要使用的子模型，可以是问题模型或上下文模型。
        ids, segments, attn_mask: 传递给子模型的输入，分别代表输入ID、段落标记和注意力掩码。
        fix_encoder: 是否固定编码器的参数。
        representation_token_pos: 表示向量的位置。
        """
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                # 根据 fix_encoder 的值确定是否在计算过程中固定模型参数。如果固定，则使用 torch.no_grad() 来禁用梯度计算，从而节省计算资源并避免更新参数
                with torch.no_grad():
                    # 使用 torch.no_grad() 上下文管理器来暂时禁用梯度计算。这意味着在此代码块内部执行的所有操作，如前向传播，都不会跟踪梯度
                    sequence_output, pooled_output, hidden_states = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                    )

                if sub_model.training:
                    # 当 requires_grad 设置为 True 时，PyTorch 会开始跟踪在这些张量上执行的所有操作，用于随后的梯度计算
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                )

        # 返回编码后的序列输出、汇总输出和隐藏状态
        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
    ) -> Tuple[T, T]:
        """ 
        包含问题和上下文的输入ID、段落标记和注意力掩码。
        encoder_type: 编码器类型，指示使用问题编码器还是上下文编码器。
        representation_token_pos: 在序列输出中表示向量的位置
        """
        q_encoder = self.question_model if encoder_type is None or encoder_type == "question" else self.ctx_model # 根据指定的 encoder_type 选择正确的编码器进行编码
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
        )

        ctx_encoder = self.ctx_model if encoder_type is None or encoder_type == "ctx" else self.question_model
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
            ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
        )

        return q_pooled_out, ctx_pooled_out # 返回问题和上下文的汇总输出

    def create_biencoder_input(
        self,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
    ) -> BiEncoderBatch:
        """
        准备双编码器训练数据的
        samples: 包含BiEncoderSample实例的列表，每个样本包含一个查询和与之相关的正面和负面上下文。
        tensorizer: 用于将文本序列转换成模型可以处理的张量格式。
        insert_title: 是否在上下文序列开始处插入标题。
        num_hard_negatives 和 num_other_negatives: 分别表示每个问题要使用的困难负样本和其他负样本的数量。
        shuffle 和 shuffle_positives: 分别表示是否打乱负样本池和正样本池。
        hard_neg_fallback: 如果没有足够的困难负样本，是否回退到使用普通负样本。
        query_token: 可选的查询令牌，用于在查询前添加特定令牌。
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # 遍历每个样本，从中提取和处理查询和上下文数据
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample.positive_passages[0]

            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages
            question = sample.query
            # question = normalize_question(sample.query)

            if shuffle:
                # 根据参数决定是否打乱正负样本序列
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            # hard_neg_fallback: 如果设置为 True 并且没有足够的困难负面上下文（hard_neg_ctxs），那么它会使用普通负面上下文（neg_ctxs）作为替代
            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            # 根据指定的数量（num_hard_negatives 和 num_other_negatives）来截取困难负面和其他负面上下文列表
            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            """ 
            all_ctxs 包含了选中的正面上下文、负面上下文和困难负面上下文。
            hard_negatives_start_idx 和 hard_negatives_end_idx 用于标记困难负面上下文在 all_ctxs 中的起始和结束位置。
            current_ctxs_len 用于记录当前上下文张量列表 ctx_tensors 的长度，以便正确记录新添加的上下文的索引
            """
            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            # 通过 tensorizer.text_to_tensor 方法，将每个上下文的文本（和可选的标题）转换为张量格式，并将转换后的张量添加到 ctx_tensors 列表中
            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(ctx.text, title=ctx.title if (insert_title and ctx.title) else None)
                for ctx in all_ctxs
            ]

            ctx_tensors.extend(sample_ctxs_tensors)
            # positive_ctx_indices 和 hard_neg_ctx_indices 分别记录了正面上下文和困难负面上下文在 ctx_tensors 列表中的位置
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            # 如果指定了 query_token，则会在查询前添加这个令牌，然后将修改后的查询转换为张量；否则，直接将原始查询文本转换为张量
            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(question, tensorizer, token_str=query_token)
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(tensorizer.text_to_tensor(" ".join([query_token, question])))
            else:
                question_tensors.append(tensorizer.text_to_tensor(question))

        # 使用 torch.cat 方法，将所有上下文和查询的张量分别合并成两个大张量：ctxs_tensor 和 questions_tensor
        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        # ctx_segments 和 question_segments 是用于区分输入中不同部分的额外张量，这里它们被初始化为零张量，大小与对应的上下文和查询张量相同
        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        # 返回一个 BiEncoderBatch 实例，其中包含了合并后的查询张量、上下文张量、段落标记以及正面和困难负面上下文的索引
        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
        )

    # 加载和保存模型状态，使模型可以从保存的状态继续训练或用于推断
    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        # saved_state: 这是一个类型为 CheckpointState 的实例，假设它包含了要加载到当前模型的状态。
        # strict: 一个布尔值，用于指定是否严格执行状态字典的加载。当 strict=True 时，保存的状态和模型状态必须完全匹配；当 strict=False 时，不匹配的参数将被忽略
        # 使用 self.load_state_dict() 方法加载 saved_state.model_dict，strict 参数控制加载过程的严格性
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        # 返回当前模型的状态字典
        return self.state_dict()


class BiEncoderNllLoss(object):
    # 计算双编码器模型在一批查询和上下文向量之间的负对数似然（Negative Log Likelihood, NLL）损失
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T, # 查询向量和上下文向量
        positive_idx_per_question: list, # 每个查询向量对应的正向上下文向量的索引
        hard_negative_idx_per_question: list = None, # 每个查询向量对应的难负样本上下文向量的索引，尽管在当前实现中未使用，但可以在将来用于损失函数的修改
        loss_scale: float = None, # 损失缩放因子，可以用来调节损失的大小
    ) -> Tuple[T, int]:
        """
        计算给定的查询向量(q_vectors)和上下文向量(ctx_vectors)列表之间的NLL损失
        """
        scores = self.get_scores(q_vectors, ctx_vectors) # 计算查询向量和上下文向量之间的相似度分数

        if len(q_vectors.size()) > 1:
            # 检查q_vectors是否包含多个维度，即是否包含多个查询向量
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1) # 对分数应用Log-Softmax函数，以便后续计算NLL损失

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        ) # 计算NLL损失，使用查询向量对应的正向上下文向量的索引作为目标

        max_score, max_idxs = torch.max(softmax_scores, 1) # 找出每个查询向量的最高分数及其索引
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum() # 计算正确预测的数量

        if loss_scale:
            # 如果指定了损失缩放因子，则调整损失值
            loss.mul_(loss_scale)

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


def _select_span_with_token(text: str, tensorizer: Tensorizer, token_str: str = "[START_ENT]") -> T:
    # 在给定文本中寻找特定的令牌（例如 "[START_ENT]"），并基于该令牌的位置调整文本，使其适合模型处理
    id = tensorizer.get_token_id(token_str) # 获取特定令牌（如 "[START_ENT]"）的ID
    query_tensor = tensorizer.text_to_tensor(text) # 将原始文本转换成模型可以处理的张量形式

    if id not in query_tensor: # 如果转换后的张量中没有该特定令牌，函数会执行一个更细致的处理流程
        # 生成一个不应用最大长度限制的完整张量，以确保特定令牌没有因长度限制而被剪切
        query_tensor_full = tensorizer.text_to_tensor(text, apply_max_len=False)
        # 找到特定令牌在张量中的所有索引
        token_indexes = (query_tensor_full == id).nonzero()
        if token_indexes.size(0) > 0: # 如果找到了特定令牌，就计算其开始位置 start_pos
            start_pos = token_indexes[0, 0].item()
            # add some randomization to avoid overfitting to a specific token position

            # 在原始令牌位置基础上，向左移动一定距离，以获取一段文本。这部分文本的长度由 tensorizer.max_length 决定，且位置有一定的随机化，以避免模型过分适应特定位置
            left_shit = int(tensorizer.max_length / 2)
            rnd_shift = int((rnd.random() - 0.5) * left_shit / 2)
            left_shit += rnd_shift

            query_tensor = query_tensor_full[start_pos - left_shit :]
            # 如果调整后的文本不是以 CLS 令牌开始的，则在前面添加一个 CLS 令牌
            cls_id = tensorizer.tokenizer.cls_token_id
            if query_tensor[0] != cls_id:
                query_tensor = torch.cat([torch.tensor([cls_id]), query_tensor], dim=0)

            from dpr.models.reader import _pad_to_len

            # 调用 _pad_to_len 函数将调整后的张量填充或修剪到模型需要的固定长度，且确保以 SEP 令牌结束
            query_tensor = _pad_to_len(query_tensor, tensorizer.get_pad_id(), tensorizer.max_length)
            query_tensor[-1] = tensorizer.tokenizer.sep_token_id
            # logger.info('aligned query_tensor %s', query_tensor)

            # 函数确保调整后的张量中包含特定的令牌。如果包含，则返回该张量；如果不包含，则抛出运行时错误
            assert id in query_tensor, "query_tensor={}".format(query_tensor)
            return query_tensor
        else:
            raise RuntimeError("[START_ENT] toke not found for Entity Linking sample query={}".format(text))
    else:
        return query_tensor
