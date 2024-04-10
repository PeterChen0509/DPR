#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib

"""
 'Router'-like set of methods for component initialization with lazy imports 
"""


def init_hf_bert_biencoder(args, **kwargs):
    # 首先检查transformers库是否已安装，如果没有安装，则抛出一个运行时错误。如果已安装，它将从hf_models模块导入get_bert_biencoder_components函数，并调用此函数来获取BERT双编码器的组件
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")
    from .hf_models import get_bert_biencoder_components

    return get_bert_biencoder_components(args, **kwargs)


def init_hf_bert_reader(args, **kwargs):
    # 初始化基于 Hugging Face transformers 库的 BERT 阅读理解模型组件
    if importlib.util.find_spec("transformers") is None:
        # 使用 importlib.util.find_spec 函数检查是否已安装 transformers 库
        raise RuntimeError("Please install transformers lib")
    from .hf_models import get_bert_reader_components # 获取设置 BERT 模型为阅读理解任务所需的各个组件

    return get_bert_reader_components(args, **kwargs)


def init_pytext_bert_biencoder(args, **kwargs):
    if importlib.util.find_spec("pytext") is None:
        raise RuntimeError("Please install pytext lib")
    from .pytext_models import get_bert_biencoder_components

    return get_bert_biencoder_components(args, **kwargs)


def init_fairseq_roberta_biencoder(args, **kwargs):
    if importlib.util.find_spec("fairseq") is None:
        raise RuntimeError("Please install fairseq lib")
    from .fairseq_models import get_roberta_biencoder_components

    return get_roberta_biencoder_components(args, **kwargs)


def init_hf_bert_tenzorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")
    from .hf_models import get_bert_tensorizer

    return get_bert_tensorizer(args)


def init_hf_roberta_tenzorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Please install transformers lib")
    from .hf_models import get_roberta_tensorizer
    return get_roberta_tensorizer(args.encoder.pretrained_model_cfg, args.do_lower_case, args.encoder.sequence_length)


BIENCODER_INITIALIZERS = {
    "hf_bert": init_hf_bert_biencoder,
    "pytext_bert": init_pytext_bert_biencoder,
    "fairseq_roberta": init_fairseq_roberta_biencoder,
}

READER_INITIALIZERS = {
    "hf_bert": init_hf_bert_reader,
}

TENSORIZER_INITIALIZERS = {
    "hf_bert": init_hf_bert_tenzorizer,
    "hf_roberta": init_hf_roberta_tenzorizer,
    "pytext_bert": init_hf_bert_tenzorizer,  # using HF's code as of now
    "fairseq_roberta": init_hf_roberta_tenzorizer,  # using HF's code as of now
}


def init_comp(initializers_dict, type, args, **kwargs):
    # 根据提供的type（模型类型），这个函数查找initializers_dict字典中相应的初始化函数，并用提供的args和kwargs调用它。如果指定的类型在字典中不存在，它会抛出一个RuntimeError异常
    if type in initializers_dict:
        return initializers_dict[type](args, **kwargs)
    else:
        raise RuntimeError("unsupported model type: {}".format(type))


def init_biencoder_components(encoder_type: str, args, **kwargs):
    # 这个函数是一个专门用于初始化双编码器（biencoder）组件的函数。它使用BIENCODER_INITIALIZERS字典（该字典需要在别处定义）来查找并初始化指定类型的编码器
    return init_comp(BIENCODER_INITIALIZERS, encoder_type, args, **kwargs)


def init_reader_components(encoder_type: str, args, **kwargs):
    # 初始化和配置阅读器组件
    return init_comp(READER_INITIALIZERS, encoder_type, args, **kwargs)


def init_tenzorizer(encoder_type: str, args, **kwargs):
    return init_comp(TENSORIZER_INITIALIZERS, encoder_type, args, **kwargs)
