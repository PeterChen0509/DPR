#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Most of the tokenizers code here is copied from DrQA codebase to avoid adding extra dependency
"""

import copy
import logging

import regex
import spacy

logger = logging.getLogger(__name__)


class Tokens(object):
    """
    Tokens 类提供了一个丰富的接口，用于访问和操作分词数据，包括但不限于提取词性标注、词元、实体标签，以及生成 n-gram
    Tokens 类定义了一系列类变量，用作分词数据字典中的键（TEXT, TEXT_WS, SPAN, POS, LEMMA, NER），分别代表文本、带空格的文本、跨度、词性标注、词元（基本形式）、命名实体识别标签
    """

    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        """ 
        data 应该是一个列表，其中每个元素都是一个字典，包含上述类变量定义的键。
        annotators 指明了哪些注释信息是可用的，如词性、词元或实体标签。
        opts 可以包含任何其他选项，用于控制类的行为。
        """
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        # 返回分词列表的长度，即文本中分词的数量
        return len(self.data)

    def slice(self, i=None, j=None):
        # 返回 Tokens 对象的一个视图，包含从索引 i 到 j 的分词子集
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i:j]
        return new_tokens

    def untokenize(self):
        # 将分词列表重新组合为原始文本，包括重新插入空格
        return "".join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        # 返回每个分词的文本列表。如果 uncased 为 True，则返回小写文本。
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        # 返回每个分词的字符偏移量列表，每个偏移量以 [start, end) 格式表示
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        # 返回每个分词的词性标签列表，如果这种注释未包含在 annotators 中，则返回 None。
        if "pos" not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        # 返回每个分词的词元文本列表，如果这种注释未包含在 annotators 中，则返回 None。
        if "lemma" not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        # 返回每个分词的命名实体识别标签列表，如果这种注释未包含在 annotators 中，则返回 None。
        if "ner" not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """
        返回从长度 1 到 n 的所有 n-gram 的列表。
        
        n: n-gram 的长度。默认值为 1，即单个词。
        uncased: 一个布尔值，指示是否将文本转换为小写。默认为 False。
        filter_fn: 一个可选的用户定义函数，用于过滤 n-gram。只有当此函数返回 False 时，对应的 n-gram 才会被保留。
        as_strings: 控制返回的 n-gram 是作为字符串还是作为词汇列表。默认为 True，返回字符串形式的 n-gram。
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased) # 获取文本的词汇列表，根据 uncased 参数决定是否转换为小写
        # 通过两层循环构造 n-gram 的索引范围。外层循环遍历词汇列表的每个起始点 s，内层循环遍历从 s 开始到 s + n（或词汇列表末尾，取二者中的较小者）的每个结束点 e。对于每一对 (s, e)，如果这个 n-gram 未被 filter_fn 函数过滤掉（即 _skip 返回 False），则将其索引范围 (s, e + 1) 添加到 ngrams 列表中
        ngrams = [
            (s, e + 1)
            for s in range(len(words))
            for e in range(s, min(s + n, len(words)))
            if not _skip(words[s : e + 1])
        ]

        # Concatenate into strings
        # 如果 as_strings 为 True，则将索引范围内的词汇连接为字符串形式的 n-gram；否则保留为索引范围
        if as_strings:
            ngrams = ["{}".format(" ".join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        # 将具有相同命名实体识别（NER）标签的连续实体令牌组合到一起。它适用于那些已经进行了实体识别处理的文本，其中每个令牌都被分配了一个NER标签（例如，人名、地点、组织等）
        entities = self.entities() # 获取文本中所有令牌的NER标签列表
        if not entities:
            return None
        # 通过 self.opts.get("non_ent", "O") 获取表示非实体的NER标签，默认为 "O"（这是在实体识别中通常用于表示“非实体”的标签）
        non_ent = self.opts.get("non_ent", "O")
        # 初始化一个空列表 groups 用于存储最终的实体组，以及一个索引变量 idx 用于遍历 entities 列表
        groups = []
        idx = 0
        # 如果当前NER标签不是非实体标签（即，当前令牌是某种实体），则记录起始位置 start，然后继续向前移动 idx，直到遇到不同的NER标签或到达列表末尾。这个过程称为“chomp the sequence”，即吞噬整个连续相同标签的序列
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                # 将从 start 到 idx 的令牌序列（通过 self.slice(start, idx).untokenize() 方法得到的字符串）和它们共享的NER标签作为一个元组添加到 groups 列表中
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                # 如果当前NER标签是非实体标签，只需将 idx 增加1，继续检查下一个令牌
                idx += 1
        # 返回一个元组列表 groups，每个元组包含两个元素：一个是组成实体组的令牌序列组成的字符串，另一个是这些令牌共享的NER标签
        return groups


class Tokenizer(object):
    """
    文本分词的类的模板
    在自然语言处理（NLP）中，分词是将文本分割成更小单元（如单词、短语或句子）的过程，这是文本预处理的关键步骤之一
    """

    def tokenize(self, text):
        # 实际的分词逻辑
        raise NotImplementedError

    def shutdown(self):
        # 执行分词器关闭前的清理工作
        pass

    def __del__(self):
        # 当对象被垃圾回收时会自动调用。在这个 Tokenizer 类中，__del__ 方法被用来在对象被销毁之前调用 shutdown 方法。这样做可以确保分词器在不再需要时能够优雅地关闭和清理资源
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    # 进行简单的文本分词
    # 使用正则表达式来匹配文本中的字母数字字符和非空格字符，生成一系列的令牌（tokens）
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+" # 匹配一个或多个字母数字字符（包括 Unicode 字母、数字、标记）
    NON_WS = r"[^\p{Z}\p{C}]"   # 匹配任何非空白字符

    def __init__(self, **kwargs):
        """
        接受任意数量的关键字参数
        Args:
            annotators: None or empty set (only tokenizes).
        """
        # 构造函数首先编译一个正则表达式，该表达式匹配 ALPHA_NUM 或 NON_WS 定义的模式。这里使用了 regex 库，而不是 Python 的内置 re 模块，因为 regex 支持更广泛的 Unicode 属性匹配
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )
        # 如果 kwargs 中包含 annotators 键，并且其值非空，将打印一条警告信息，说明 SimpleTokenizer 只支持分词操作，不支持其他注解类型
        if len(kwargs.get("annotators", {})) > 0:
            logger.warning(
                "%s only tokenizes! Skipping annotators: %s" % (type(self).__name__, kwargs.get("annotators"))
            )
        # 初始化一个空的 annotators 集合
        self.annotators = set()

    def tokenize(self, text):
        # 接受一个字符串 text 作为输入，并返回一个 Tokens 类的实例
        data = [] # 创建一个空列表 data 来存储结果
        matches = [m for m in self._regexp.finditer(text)] # 使用先前编译的正则表达式 _regexp 来查找 text 中所有匹配的序列
        for i in range(len(matches)):
            # Get text
            token = matches[i].group() # 获取匹配的文本，即令牌

            # Get whitespace
            # 获取匹配文本在原始文本中的位置（span）
            span = matches[i].span()
            start_ws = span[0] # 匹配文本的开始位置
            if i + 1 < len(matches):
                # 如果当前匹配项不是最后一个，那么下一个令牌的开始位置 matches[i + 1].span()[0] 用作当前令牌后的空白字符的结束位置（end_ws）。这样可以计算出位于两个令牌之间的空白字符
                end_ws = matches[i + 1].span()[0]
            else:
                # 如果是最后一个匹配项，则当前令牌的结束位置也是空白字符的结束位置
                end_ws = span[1]

            # Format data
            # 每个令牌和其后的空白区域被作为一个元组添加到 data 列表中
            data.append(
                (
                    token,
                    text[start_ws:end_ws],
                    span,
                )
            )
        """ 
        text = "Hello, world!"
        我们获取每个单词的文本和在原始字符串中的位置。
        然后，对于 "Hello"，我们还计算出它和下一个单词 "world" 之间的空白字符（包括逗号和空格）。
        对于最后一个单词 "world"，我们只考虑它后面直到文本结束的部分
        """
        return Tokens(data, self.annotators)


class SpacyTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = kwargs.get("model", "en_core_web_sm")  # TODO: replace with en ?
        self.annotators = copy.deepcopy(kwargs.get("annotators", set()))
        nlp_kwargs = {"parser": False}
        if not any([p in self.annotators for p in ["lemma", "pos", "ner"]]):
            nlp_kwargs["tagger"] = False
        if "ner" not in self.annotators:
            nlp_kwargs["entity"] = False
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace("\n", " ")
        tokens = self.nlp.tokenizer(clean_text)
        if any([p in self.annotators for p in ["lemma", "pos", "ner"]]):
            self.nlp.tagger(tokens)
        if "ner" in self.annotators:
            self.nlp.entity(tokens)

        data = []
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i].idx
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1].idx
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)

            data.append(
                (
                    tokens[i].text,
                    text[start_ws:end_ws],
                    (tokens[i].idx, tokens[i].idx + len(tokens[i].text)),
                    tokens[i].tag_,
                    tokens[i].lemma_,
                    tokens[i].ent_type_,
                )
            )

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={"non_ent": ""})
