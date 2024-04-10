#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Set of utilities for Q&A results validation tasks - Retriver passage validation and Reader predicted answer validation
"""

import collections
import logging
import string
import unicodedata
import zlib
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict

import regex as re

from dpr.data.retriever_data import TableChunk
from dpr.utils.tokenizers import SimpleTokenizer

logger = logging.getLogger(__name__)

QAMatchStats = collections.namedtuple("QAMatchStats", ["top_k_hits", "questions_doc_hits"])

QATableMatchStats = collections.namedtuple(
    "QAMatchStats", ["top_k_chunk_hits", "top_k_table_hits", "questions_doc_hits"]
)


def calculate_matches(
    all_docs: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    closest_docs: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> QAMatchStats:
    """
    评估一组文档中是否存在预定的答案
    all_docs: 一个字典，包含整个文档数据库，其中doc_id映射到一个包含文档文本和标题的元组。
    answers: 一个答案列表的列表，每个列表对应一个问题的答案。
    closest_docs: 最佳结果文档的ID及其分数的列表，每个元素是一个元组，其中包含文档ID列表和分数列表。
    workers_num: 用于处理数据的并行线程数量。
    match_type: 答案匹配的类型，具体选项请参考has_answer函数的代码。
    """
    logger.info("all_docs size %d", len(all_docs))
    # 记录所有文档的大小，并将all_docs赋值给全局变量dpr_all_documents，用于在多个进程中共享
    global dpr_all_documents
    dpr_all_documents = all_docs
    logger.info("dpr_all_documents size %d", len(dpr_all_documents))

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts) # 实例化一个简单的分词器SimpleTokenizer

    # 创建一个ProcessPool实例来管理子进程，数量由workers_num参数指定
    processes = ProcessPool(processes=workers_num)
    logger.info("Matching answers in top docs...")
    # 使用partial函数固定check_answer函数的match_type和tokenizer参数，创建一个偏函数get_score_partial
    get_score_partial = partial(check_answer, match_type=match_type, tokenizer=tokenizer)

    # 将answers和closest_docs打包(zip)成一个元组的列表，每个元组包含一个问题的答案列表和对应的最接近的文档
    questions_answers_docs = zip(answers, closest_docs)
    # 使用进程池processes映射get_score_partial到每个questions_answers_docs元组，以并行方式计算每个问题的答案匹配得分
    scores = processes.map(get_score_partial, questions_answers_docs)

    logger.info("Per question validation results len=%d", len(scores))

    n_docs = len(closest_docs[0][0]) # 每个问题的顶部文档数量n_docs，假设每个问题返回的顶部文档数量是相同的，所以这里取第一个问题的顶部文档数量作为n_docs
    top_k_hits = [0] * n_docs # 初始化一个长度为n_docs的列表top_k_hits，所有元素都设为0。这个列表用于记录在不同的顶部k文档中找到有效匹配的总次数
    for question_hits in scores:
        # 计算每个问题在顶部n个文档中的最佳匹配位置（如果存在）
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            # 从best_hit索引开始到列表末尾，将top_k_hits中的每个值增加1。这表示对于当前问题，从找到第一个匹配的文档开始，所有后续的top_k统计都应该计入一个有效匹配
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    # top_k_hits: 一个列表，其索引是检索的顶部文档数量，值是整个数据集中有效匹配的总数。
    # scores: 对于每个问题和每个检索到的文档，更详细的答案匹配信息。
    return QAMatchStats(top_k_hits, scores)


def calculate_matches_from_meta(
    answers: List[List[str]],
    closest_docs: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
    use_title: bool = False,
    meta_compressed: bool = False,
) -> QAMatchStats:
    # 通过元数据来计算问题的答案与最接近的文档之间的匹配程度
    """ 
    该函数采用多进程的方式来提高处理效率，并返回一个 QAMatchStats 对象，其中包含了匹配统计信息
    
    接受六个参数，包括答案列表 answers、最接近文档的列表 closest_docs、工作线程数 workers_num、匹配类型 match_type，以及两个可选参数 use_title（是否使用文档标题进行匹配）和 meta_compressed（元数据是否被压缩）
    """

    # 创建一个空字典 tok_opts，并使用它作为参数初始化 SimpleTokenizer 分词器。这里假设 SimpleTokenizer 是一个简单的文本分词器，用于文本的预处理
    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    # 使用 workers_num 初始化一个进程池 ProcessPool，以并行处理数据
    processes = ProcessPool(processes=workers_num)
    logger.info("Matching answers in top docs...")
    # 使用 functools.partial 创建 check_answer_from_meta 函数的部分应用版本 get_score_partial。这意味着为该函数预设了几个参数值，以便后续调用时只需提供剩余参数
    get_score_partial = partial(
        check_answer_from_meta,
        match_type=match_type,
        tokenizer=tokenizer,
        use_title=use_title,
        meta_compressed=meta_compressed,
    )

    # 使用 zip 函数将 answers 和 closest_docs 打包成一系列的元组，每个元组包含一组答案和相应的最接近文档及其分数
    questions_answers_docs = zip(answers, closest_docs)
    # 通过进程池的 map 方法，将 get_score_partial 函数应用于每个问题与答案的元组上，计算每个问题的匹配分数
    scores = processes.map(get_score_partial, questions_answers_docs)

    # 记录每个问题验证结果的数量
    logger.info("Per question validation results len=%d", len(scores))

    # 基于 closest_docs 中第一个元素的长度（即文档数量），初始化一个记录每个文档命中次数的列表 top_k_hits
    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs
    # 遍历每个问题的匹配分数，找到每个问题的最佳匹配文档（如果存在），并更新 top_k_hits 列表来反映这些最佳匹配文档及其后续文档的命中次数增加
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    # 使用计算出的顶部 K 文档命中次数和每个问题的分数列表创建 QAMatchStats 对象并返回
    return QAMatchStats(top_k_hits, scores)


def check_answer(questions_answers_docs, tokenizer, match_type) -> List[bool]:
    """
    检查一组顶部文档是否包含了给定问题的答案
    返回一个布尔值列表，表示每个顶部文档是否包含至少一个答案
    
    questions_answers_docs: 一个元组，其中包含一组答案和与之关联的顶部文档的ID和得分。
    tokenizer: 用于文本分词的工具。
    match_type: 答案匹配类型的标识符，用于指示如何匹配答案和文档中的文本。
    """
    # 解构questions_answers_docs元组以获取答案列表answers和顶部文档的ID及其得分的元组(doc_ids, doc_scores)
    answers, (doc_ids, doc_scores) = questions_answers_docs

    global dpr_all_documents
    hits = [] # 存储每个顶部文档是否包含答案的布尔值结果

    for i, doc_id in enumerate(doc_ids): # 遍历每个顶部文档的ID
        doc = dpr_all_documents[doc_id] # 通过文档ID从全局字典dpr_all_documents中检索文档，该字典包含了所有文档的文本和标题
        text = doc[0]

        answer_found = False
        if text is None:  
            # 如果由于某种原因无法找到文档（text为None），记录一条警告日志，并将False添加到hits列表中，然后继续处理下一个文档
            logger.warning("no doc in db")
            hits.append(False)
            continue
        # 根据match_type的值，使用不同的函数来检查文档文本中是否存在任一答案
        if match_type == "kilt":
            if has_answer_kilt(answers, text):
                answer_found = True
        elif has_answer(answers, text, tokenizer, match_type):
            answer_found = True
        # 根据答案是否被找到（answer_found），将相应的布尔值添加到hits列表中
        hits.append(answer_found)
    return hits


def check_answer_from_meta(
    questions_answers_docs,
    tokenizer,
    match_type,
    meta_body_idx: int = 1,
    meta_title_idx: int = 2,
    use_title: bool = False,
    meta_compressed: bool = False,
) -> List[bool]:
    """
    遍历一系列文档的元数据，检查这些文档中是否包含给定的答案。它可以处理文档正文和标题，并支持压缩格式的文档内容。这个函数特别适用于信息检索和问答系统，其中你需要验证一个文档集合是否包含对某些问题的答案
    
    questions_answers_docs：包含问题、答案和文档元数据的元组。这个参数预期是一个复杂的数据结构，其中包含了要检查的答案，以及相关文档的元数据和得分。
    tokenizer：用于文本处理的分词器实例。
    match_type：指定答案匹配类型，可以是基于字符串的匹配或基于正则表达式的匹配。
    meta_body_idx：指定文档正文在元数据中的索引位置。默认为 1。
    meta_title_idx：指定文档标题在元数据中的索引位置。默认为 2。
    use_title：布尔值，指示是否在搜索答案时包含文档的标题。
    meta_compressed：布尔值，指示元数据是否被压缩。如果为真，函数会在处理之前解压文档的正文和标题。
    """
    # 从 questions_answers_docs 解包获取答案列表和文档元数据（docs_meta）及其得分（doc_scores）
    answers, (docs_meta, doc_scores) = questions_answers_docs

    hits = []   # 记录每个文档是否包含至少一个答案

    for i, doc_meta in enumerate(docs_meta): # 函数逐个处理每个文档的元数据

        # 提取文档的正文和标题
        text = doc_meta[meta_body_idx]
        title = doc_meta[meta_title_idx] if len(doc_meta) > meta_title_idx else ""
        if meta_compressed:
            # 如果元数据被压缩（meta_compressed 为 True），则先解压
            text = zlib.decompress(text).decode()
            title = zlib.decompress(title).decode()

        if use_title:
            # 将标题和正文合并为一个文本字符串
            text = title + " . " + text
        answer_found = False
        if has_answer(answers, text, tokenizer, match_type):
            # 调用 has_answer 函数检查合并后的文本中是否包含任何给定的答案。如果找到答案，answer_found 标志被设置为 True
            answer_found = True
        hits.append(answer_found)
    # 函数返回 hits 列表，其中每个元素对应于 docs_meta 中的一个文档，表示该文档是否包含答案
    return hits


def has_answer(answers, text, tokenizer, match_type) -> bool:
    """
    检查一个文档（text）是否包含给定答案（answers）中的任一项。根据match_type参数的值，这个检查可以是字符串匹配或正则表达式匹配
    
    answers：可能的答案列表，可以是字符串列表或正则表达式列表。
    text：要搜索的文本。
    tokenizer：一个用于文本分词的对象。
    match_type：一个字符串，指示匹配类型。如果是"string"，将使用字符串匹配；如果是"regex"，将使用正则表达式匹配。
    """
    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True) # 文本被分词，并转换为不区分大小写的形式

        for single_answer in answers:
            # 遍历answers中的每一个答案，对每个答案进行规范化、分词，并转换为不区分大小写的单词列表
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            # 使用嵌套循环，函数尝试在文本的每个可能的位置查找答案的匹配项。如果找到匹配，函数立即返回True
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            # 尝试在整个文本中使用正则表达式搜索答案。如果找到匹配，函数返回True
            if regex_match(text, single_answer):
                # 如果在文本中找到至少一个答案，函数返回True；否则，返回False
                return True
    return False


def regex_match(text, pattern):
    # 检测一个给定的正则表达式模式是否在某段文本中
    """ 
    text：一个字符串，表示需要搜索的文本。
    pattern：一个字符串，表示正则表达式模式。
    
    re.IGNORECASE：这个标志表示在匹配时忽略大小写。
    re.UNICODE：这个标志使正则表达式支持Unicode字符匹配。
    re.MULTILINE：这个标志表示多行模式，影响^和$的行为。在多行模式下，^匹配字符串的开始和每行的开始，$匹配字符串的结束和每行的结束。
    """
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        # 如果正则表达式有误，函数不会引发错误，而是简单地指示模式没有找到
        return False
    # search方法会在text中搜索第一个与正则表达式匹配的位置。如果找到匹配，search方法返回一个匹配对象；如果找到匹配，则返回True；如果没有找到，则返回False
    return pattern.search(text) is not None


# function for the reader model answer validation
def exact_match_score(prediction, ground_truth):
    """ 
    评估预测答案与真实答案是否完全一致（在规范化后）
    prediction：模型预测的答案。
    ground_truth：真实的答案。
    返回值：如果规范化后的预测答案与真实答案完全一致，则返回True；否则返回False。
    """
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _normalize_answer(s):
    # 对给定的文本字符串进行规范化处理，以便进行准确的匹配比较
    def remove_articles(text):
        # 去除英文中的冠词（"a", "an", "the"），因为它们通常不会影响答案的正确性
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        # 通过分割和重新连接字符串来去除多余的空格
        return " ".join(text.split())

    def remove_punc(text):
        # 移除所有的标点符号，因为标点通常不影响答案的含义
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        # 将所有字符转换为小写，以忽略大小写差异
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _normalize(text):
    # 使用 Python 的 unicodedata 模块来规范化文本。规范化是将文本转换为一致格式的过程，以减少编码上的差异，特别是对于包含特殊字符和符号的文本。这在处理多语言数据时尤其重要，因为不同的字符和符号可能有多种不同的编码表示。
    # "NFD" 是规范化的形式之一，代表 Normalization Form Decomposition。这种形式将字符分解为基字符和组合标记。例如，字符 'é'（e with acute accent）会被分解成 'e' 和一个单独的重音符号
    return unicodedata.normalize("NFD", text)


def calculate_chunked_matches(
    all_docs: Dict[object, TableChunk],
    answers: List[List[str]],
    closest_docs: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> QATableMatchStats:
    """ 
    计算和统计问答系统中，答案与一组文档中的信息块（如表格块）的匹配程度。该函数采用多进程方式处理大量数据，以提高处理效率
    
    all_docs: 一个字典，键是文档或表格的ID，值是 TableChunk 对象，包含表格的字符串表示、标题和ID。
    answers: 问题的答案列表，每个元素也是一个答案列表，表示可能的多个答案。
    closest_docs: 最接近的文档列表，每个元素是一个元组，包含文档ID列表和相应的分数列表。
    workers_num: 工作进程的数量，用于并行处理。
    match_type: 匹配类型，例如“精确匹配”或“模糊匹配”。
    返回 QATableMatchStats，一个包含匹配统计数据的对象。
    """
    # 存储所有文档和表格的信息
    global dpr_all_documents
    dpr_all_documents = all_docs

    global dpr_all_tables
    dpr_all_tables = {}

    # 遍历 all_docs，将表格按ID组织，以便后续处理。
    for key, table_chunk in all_docs.items():
        table_str, title, table_id = table_chunk
        table_chunks = dpr_all_tables.get(table_id, [])
        table_chunks.append((table_str, title))
        dpr_all_tables[table_id] = table_chunks

    tok_opts = {}
    # 创建一个简单的分词器 tokenizer，用于处理文档和问题文本
    tokenizer = SimpleTokenizer(**tok_opts)

    # 创建一个进程池，以并行方式计算答案和文档块的匹配度
    processes = ProcessPool(processes=workers_num)

    logger.info("Matching answers in top docs...")
    # 定义部分函数 get_score_partial，并将其与每个问题的答案和相关文档配对，使用多进程映射（map）进行处理
    get_score_partial = partial(check_chunked_docs_answer, match_type=match_type, tokenizer=tokenizer)
    questions_answers_docs = zip(answers, closest_docs)
    scores = processes.map(get_score_partial, questions_answers_docs)
    logger.info("Per question validation results len=%d", len(scores))

    
    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs
    top_k_orig_hits = [0] * n_docs
    # # 根据处理结果更新匹配统计，包括 top_k_hits 和 top_k_orig_hits，分别代表在前K个文档中找到答案的次数和原始文档中找到答案的次数
    for s in scores:
        question_hits, question_orig_doc_hits = s
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

        best_hit = next((i for i, x in enumerate(question_orig_doc_hits) if x), None)
        if best_hit is not None:
            top_k_orig_hits[best_hit:] = [v + 1 for v in top_k_orig_hits[best_hit:]]

    return QATableMatchStats(top_k_hits, top_k_orig_hits, scores)


# -------------------- KILT eval ---------------------------------


def has_answer_kilt(answers, text) -> bool:
    # 检查一个给定的文本 (text) 中是否包含列表 (answers) 中的任何一个答案
    """ 
    这个函数接受两个参数：answers（一个答案的列表）和 text（需要搜索的文本）。
    首先，使用 normalize_kilt 函数对 text 进行规范化处理。
    然后，遍历 answers 列表中的每个答案：
    对每个答案也使用 normalize_kilt 函数进行规范化处理。
    检查规范化后的答案是否为规范化文本的子字符串。
    如果找到匹配项，则函数返回 True。
    如果遍历完所有答案都没有找到匹配项，则函数返回 False。
    """
    text = normalize_kilt(text)
    for single_answer in answers:
        single_answer = normalize_kilt(single_answer)
        if single_answer in text:
            return True
    return False


# answer normalization
def normalize_kilt(s):
    # 对输入的字符串 s 进行规范化处理，包括转换为小写、移除标点符号、去除冠词以及多余的空格，最终返回处理后的字符串
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        # 使用正则表达式 \b(a|an|the)\b 匹配文本中的 a、an 和 the，并将它们替换为空格
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        # 将文本中的多个空格替换为一个空格，以去除多余的空格
        return " ".join(text.split())

    def remove_punc(text):
        # 遍历字符串，移除所有的标点符号字符
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
