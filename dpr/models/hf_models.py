#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple, List

import torch
import transformers
from torch import Tensor as T
from torch import nn


if transformers.__version__.startswith("4"):
    from transformers import BertConfig, BertModel
    from transformers import AdamW
    from transformers import BertTokenizer
    from transformers import RobertaTokenizer
else:
    from transformers.modeling_bert import BertConfig, BertModel
    from transformers.optimization import AdamW
    from transformers.tokenization_bert import BertTokenizer
    from transformers.tokenization_roberta import RobertaTokenizer

from dpr.utils.data_utils import Tensorizer
from dpr.models.biencoder import BiEncoder
from .reader import Reader

logger = logging.getLogger(__name__)


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    # 如果cfg.encoder具有dropout属性，则使用该值；否则，默认为0.0
    # hasattr 是 Python 中的一个内置函数，用于判断一个对象是否具有指定的属性或方法
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    # 初始化问题编码器（question_encoder）和上下文编码器（ctx_encoder）
    question_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg, # 预训练模型的配置
        projection_dim=cfg.encoder.projection_dim, # 投影维度
        dropout=dropout, # 上一步中设置的dropout值
        pretrained=cfg.encoder.pretrained, # 指示是否使用预训练权重
        **kwargs
    )
    ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    # 检查配置中是否指定了fix_ctx_encoder属性。如果指定了，使用该值；如果没有指定，默认为False。这个设置用于决定是否在训练过程中固定上下文编码器的参数
    fix_ctx_encoder = cfg.encoder.fix_ctx_encoder if hasattr(cfg.encoder, "fix_ctx_encoder") else False
    # 使用上面初始化的question_encoder和ctx_encoder创建BiEncoder实例。如果fix_ctx_encoder为True，则在训练过程中不会更新上下文编码器的参数
    biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)

    # 如果inference_only为False，则会初始化一个优化器，它使用get_optimizer函数和从配置获取的参数（学习率、Adam epsilon、权重衰减）来配置。如果inference_only为True，则优化器设为None
    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    # 使用get_bert_tensorizer函数和配置信息初始化一个用于处理输入数据的张量化器
    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


def get_bert_reader_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs
    )

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, reader, optimizer


# TODO: unify tensorizer init methods
def get_bert_tensorizer(cfg):
    """ 
    初始化一个名为 BertTensorizer 的对象。BertTensorizer 是一个处理文本输入，将其转换为模型可以理解的形式（例如，转换为 token ID 序列）的工具
    """
    # 从配置中获取序列长度。这个长度用于确定模型处理的文本序列的最大长度
    sequence_length = cfg.encoder.sequence_length
    # 加载正确的模型架构和预训练权重
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg
    # 初始化一个 BERT tokenizer。这个 tokenizer 负责将文本字符串分割成 tokens，这些 tokens 可以被模型理解和处理。参数 do_lower_case 控制是否将文本转换为小写，这取决于预训练模型是如何训练的
    tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=cfg.do_lower_case)
    if cfg.special_tokens:
        # 如果有指定特殊 token，就调用 _add_special_tokens 函数将这些 token 添加到 tokenizer 的词汇表中。特殊 token 通常用于特定的模型功能，如序列的开始和结束
        _add_special_tokens(tokenizer, cfg.special_tokens)

    # 使用初始化好的 tokenizer 和指定的 sequence_length 创建 BertTensorizer 对象，并将其返回。BertTensorizer 用于将文本转换为模型可以处理的格式，包括 token ID 序列、注意力掩码等
    return BertTensorizer(tokenizer, sequence_length)


def get_bert_tensorizer_p(
    pretrained_model_cfg: str, sequence_length: int, do_lower_case: bool = True, special_tokens: List[str] = []
):
    tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=do_lower_case)
    if special_tokens:
        _add_special_tokens(tokenizer, special_tokens)
    return BertTensorizer(tokenizer, sequence_length)


def _add_special_tokens(tokenizer, special_tokens):
    # 在一个给定的分词器（tokenizer）中添加特殊令牌（special tokens）
    logger.info("Adding special tokens %s", special_tokens)
    logger.info("Tokenizer: %s", type(tokenizer)) 
    special_tokens_num = len(special_tokens) # 特殊令牌的数量
    # TODO: this is a hack-y logic that uses some private tokenizer structure which can be changed in HF code

    assert special_tokens_num < 500
    # 生成一个列表 unused_ids，包含了将要被替换为特殊令牌的 unused 令牌的ID。这里假设分词器的词汇表中已经有了一些标记为 [unused{数字}] 的占位符令牌
    unused_ids = [tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)]
    logger.info("Utilizing the following unused token ids %s", unused_ids)

    for idx, id in enumerate(unused_ids):
        # 循环通过 unused_ids 列表，为每个特殊令牌找到一个占位符令牌并替换它
        # 对于每个索引 idx 和ID id，它首先删除词汇表中的旧令牌（如 [unused0]），然后用新的特殊令牌替换它，并更新ID映射
        old_token = "[unused{}]".format(idx)
        del tokenizer.vocab[old_token]
        new_token = special_tokens[idx]
        tokenizer.vocab[new_token] = id
        tokenizer.ids_to_tokens[id] = new_token
        logging.debug("new token %s id=%s", new_token, id)

    # 将传入的特殊令牌列表直接赋值给分词器的 additional_special_tokens 属性。这样做是为了确保分词器知道这些新加入的特殊令牌
    tokenizer.additional_special_tokens = list(special_tokens)
    # 显示添加的特殊令牌列表
    logger.info("additional_special_tokens %s", tokenizer.additional_special_tokens)
    # 显示分词器中所有特殊令牌的扩展列表
    logger.info("all_special_tokens_extended: %s", tokenizer.all_special_tokens_extended)
    # 显示添加的特殊令牌的ID列表
    logger.info("additional_special_tokens_ids: %s", tokenizer.additional_special_tokens_ids)
    # 显示分词器中所有特殊令牌的列表
    logger.info("all_special_tokens %s", tokenizer.all_special_tokens)


def get_roberta_tensorizer(pretrained_model_cfg: str, do_lower_case: bool, sequence_length: int):
    tokenizer = get_roberta_tokenizer(pretrained_model_cfg, do_lower_case=do_lower_case)
    return RobertaTensorizer(tokenizer, sequence_length)


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    # 创建一个针对特定模型的优化器
    """
    输入：
    model: 需要优化的模型（一个nn.Module对象）。
    learning_rate: 学习率，控制优化器在每次更新时应用于参数的步长大小。
    adam_eps: 用于优化器的epsilon值，可以防止除以零错误。
    weight_decay: 权重衰减（L2正则化），用于防止过拟合。
    返回：
    一个配置好的优化器对象（在这里是 AdamW） 
    """
    optimizer_grouped_parameters = get_hf_model_param_grouping(model, weight_decay)
    return get_optimizer_grouped(optimizer_grouped_parameters, learning_rate, adam_eps)


def get_hf_model_param_grouping(
    model: nn.Module,
    weight_decay: float = 0.0,
):
    # 为模型参数创建分组，以便在优化时对不同类型的参数应用不同的设置（例如，不对某些参数应用权重衰减）
    no_decay = ["bias", "LayerNorm.weight"]

    return [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


def get_optimizer_grouped(
    optimizer_grouped_parameters: List,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
) -> torch.optim.Optimizer:
    # 基于分组的参数和其他设置创建优化器

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # 获取预训练的BERT模型的分词器
    # pretrained_cfg_name: 预训练模型的名称或路径。
    # do_lower_case: 是否将文本转换为小写（取决于预训练模型的要求）
    return BertTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)


def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return RobertaTokenizer.from_pretrained(pretrained_cfg_name, do_lower_case=do_lower_case)


class HFBertEncoder(BertModel):
    # config: BERT模型的配置对象。
    # project_dim: 投影维度，如果设置为非零值，则将在编码器输出上应用一个线性映射
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config) # 调用基类 BertModel 的构造函数
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero" # 确认 hidden_size（隐藏层的大小）大于0
        # 如果 project_dim 不为零，创建一个线性层 self.encode_proj 来将编码器的输出从 hidden_size 映射到 project_dim。如果 project_dim 为零，则不创建映射
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights() # 初始化模型的权重

    # 使用 @classmethod，你可以不需要创建类的实例就直接调用该方法
    @classmethod
    def init_encoder(
        cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, pretrained: bool = True, **kwargs
    ) -> BertModel:
        """
        cfg_name: 配置名或路径，用于加载预训练的BERT模型配置。
        projection_dim: 输出的投影维度。
        dropout: 用于模型的注意力和隐藏层的dropout率。
        pretrained: 指示是否加载预训练的BERT模型。
        **kwargs: 其他传递给 from_pretrained 或 HFBertEncoder 构造函数的参数。 
        """
        logger.info("Initializing HF BERT Encoder. cfg_name=%s", cfg_name)
        # 从 cfg_name 加载 BERT 配置，如果 cfg_name 为空则默认加载 'bert-base-uncased'
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        # 如果 pretrained 为真，使用更新的配置和其他参数从预训练的模型中加载 HFBertEncoder。否则，使用给定配置和投影维度初始化一个新的 HFBertEncoder 实例
        if pretrained:
            return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    def forward(
        # input_ids, token_type_ids, attention_mask: BERT模型的标准输入。
        # representation_token_pos: 指定如何从序列输出中选择表示向量，默认为0，即选择序列的第一个标记
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:

        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ) # 使用超类的 forward 方法处理输入，获取模型的输出

        # 根据 transformers 版本和配置，处理不同的输出格式，得到序列输出（sequence_output）和隐藏状态（hidden_states）
        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ):
            sequence_output = out.last_hidden_state
            pooled_output = None
            hidden_states = out.hidden_states

        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out
        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        # 根据 representation_token_pos，从序列输出中选择或计算汇总（pooled）输出
        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert representation_token_pos.size(0) == bsz, "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack([sequence_output[i, representation_token_pos[i, 1], :] for i in range(bsz)])

        # 如果定义了 encode_proj，则对汇总输出应用线性映射
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    # TODO: make a super class for all encoders
    # 返回编码器输出的维度，这取决于是否应用了线性投影。如果应用了投影，则为 project_dim，否则为配置中的 hidden_size
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BertTensorizer(Tensorizer):
    def __init__(self, tokenizer: BertTokenizer, max_length: int, pad_to_max: bool = True):
        """ 
        tokenizer: 一个 BertTokenizer 实例，用于文本的分词和编码。
        max_length: 序列的最大长度。
        pad_to_max: 布尔值，指示是否将所有序列填充到 max_length 指定的长度。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        """
        将文本（可选地带有标题）转换成BERT模型的输入张量。
        text: 待转换的文本。
        title: 可选的标题文本。
        add_special_tokens: 是否添加特殊令牌，如 [CLS] 和 [SEP]。
        apply_max_len: 是否应用最大长度限制。 
        """
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        # 如果提供了标题，它会和文本一起被编码，并由分隔符分开
        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )

        # 根据是否需要填充或截断，调整序列长度
        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) >= seq_len:
            token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids) # 返回一个包含编码后令牌ID的PyTorch张量

    def get_pair_separator_ids(self) -> T:
        # 返回一个表示分隔符（[SEP]）令牌ID的张量，用于分隔两个句子或文本片段
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        # 返回填充令牌（[PAD]）的ID
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        # 根据输入令牌张量生成注意力掩码，指示哪些令牌是有效的（非填充）
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        # 检查给定的令牌ID是否对应于子词（在BERT中，子词通常以 ## 开头）
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        # 将一系列令牌ID解码回文本字符串
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        # 设置是否将序列填充到最大长度
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        # 根据给定的令牌（文本）返回其ID
        return self.tokenizer.vocab[token]


class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(tokenizer, max_length, pad_to_max=pad_to_max)
