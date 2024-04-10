#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import glob
import json
import logging
import pickle
import time
import zlib
from typing import List, Tuple, Dict, Iterator

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn

from dpr.utils.data_utils import RepTokenSelector
from dpr.data.qa_validation import calculate_matches, calculate_chunked_matches, calculate_matches_from_meta
from dpr.data.retriever_data import KiltCsvCtxSrc, TableChunk
from dpr.indexer.faiss_indexers import (
    DenseIndexer,
)
from dpr.models import init_biencoder_components
from dpr.models.biencoder import (
    BiEncoder,
    _select_span_with_token,
)
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint

logger = logging.getLogger()
setup_logger(logger)


def generate_question_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    query_token: str = None,
    selector: RepTokenSelector = None,
) -> T:
    # 使用一个问题编码器（question_encoder）和文本转换器（tensorizer）来处理一系列问题（questions），并生成这些问题的向量表示
    # 这个过程分批进行，每批处理的问题数由参数 bsz（批大小）决定。函数最终返回所有问题的向量表示
    """ 
    question_encoder: 一个 torch.nn.Module 实例，用于编码文本到向量。
    tensorizer: 一个 Tensorizer 实例，用于将文本转换成模型可以处理的张量形式。
    questions: 一个字符串列表，包含了需要被编码的问题。
    bsz: 整数，指定了每个批次处理的问题数量。
    query_token: （可选）一个字符串，可以添加到每个问题的开头，用于特殊的查询处理。
    selector: （可选）一个 RepTokenSelector 实例，用于选择问题中哪些词的表示最应该被用来生成问题的向量表示。
    """
    n = len(questions)
    query_vectors = []  # 存储所有问题的向量表示

    with torch.no_grad():
        # 禁用梯度计算，因为在生成向量表示时不需要进行反向传播
        for j, batch_start in enumerate(range(0, n, bsz)):
            # 通过循环，以 bsz 为批大小将问题分批处理。
            batch_questions = questions[batch_start : batch_start + bsz]

            # 如果 query_token 被指定，根据其值处理每个问题文本，以生成或修改张量
            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    # 找到与 query_token 相关的特定部分，并只对这部分内容进行处理, 选出文本中某个标记（token）开始的片段
                    batch_tensors = [
                        _select_span_with_token(q, tensorizer, token_str=query_token) for q in batch_questions
                    ]
                else:
                    # 将 query_token 与每个问题文本 q 结合，并用 tensorizer.text_to_tensor 方法将合并后的文本转换为张量。这意味着每个问题文本前都会加上这个 query_token，可能是为了标识这些问题文本的特殊用途或来源
                    batch_tensors = [tensorizer.text_to_tensor(" ".join([query_token, q])) for q in batch_questions]
            elif isinstance(batch_questions[0], T):
                # 如果 batch_questions 的第一个元素是 T 类型（这里 T 没有在代码片段中定义，但通常指代一个特定的数据类型，可能是张量），则直接使用这些问题文本作为 batch_tensors，不进行任何转换
                batch_tensors = [q for q in batch_questions]
            else:
                # 使用 tensorizer.text_to_tensor 方法将每个问题文本 q 转换为张量
                batch_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]

            # 当前批次中所有张量的最大和最小长度。如果这些长度不一致（即 max_vector_len != min_vector_len），则需要对张量进行填充，以确保它们具有相同的长度，从而可以作为一个批次一起处理
            max_vector_len = max(q_t.size(1) for q_t in batch_tensors)
            min_vector_len = min(q_t.size(1) for q_t in batch_tensors)

            if max_vector_len != min_vector_len:
                # 如果问题张量的长度不同，需要将它们填充到相同的长度
                # TODO: _pad_to_len move to utils
                from dpr.models.reader import _pad_to_len
                batch_tensors = [_pad_to_len(q.squeeze(0), 0, max_vector_len) for q in batch_tensors]

            # 将问题张量堆叠成一个批次并移动到 GPU 上（如果可用）
            # 生成对应的段标记和注意力掩码
            q_ids_batch = torch.stack(batch_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            # 如果提供了 selector，则使用它来确定每个问题中代表性词的位置
            if selector:
                # 根据批次中的问题 ID 和 tensorizer 对象来确定这些位置
                rep_positions = selector.get_positions(q_ids_batch, tensorizer)

                # 根据这些代表性位置来获取问题的向量表示
                _, out, _ = BiEncoder.get_representation(
                    question_encoder,
                    q_ids_batch,
                    q_seg_batch,
                    q_attn_mask,
                    representation_token_pos=rep_positions,
                )
            else:
                # 如果没有提供 selector 对象，则直接使用 question_encoder 来获取整个问题文本的向量表示，不基于特定的代表性词位置
                _, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            # 将输出向量从 GPU 转移到 CPU，并按照第一个维度（每个问题的向量表示）进行分割，以便将这些向量逐个添加到 query_vectors 列表中
            query_vectors.extend(out.cpu().split(1, dim=0))

            # 如果 query_vectors 的长度是 100 的倍数，这一情况会被记录下来，作为编码过程的进度指示
            if len(query_vectors) % 100 == 0:
                logger.info("Encoded queries %d", len(query_vectors))

    # 将所有分离的问题向量合并为一个连续的张量 query_tensor，这样就得到了所有问题的向量表示集合
    query_tensor = torch.cat(query_vectors, dim=0)
    # 通过日志记录 query_tensor 的大小，以验证整个编码过程的结果
    logger.info("Total encoded queries tensor %s", query_tensor.size())
    # 确保最终的 query_tensor 包含的向量数量与原始问题列表中的问题数量相匹配，这是一个正确性检查
    assert query_tensor.size(0) == len(questions)
    return query_tensor  # 返回包含所有问题向量表示的 query_tensor


class DenseRetriever(object):
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer):
        """ 
        question_encoder: 一个神经网络模块，用于将文本问题编码成向量。此参数的类型应为 nn.Module，表明它是一个 PyTorch 神经网络模块。
        batch_size: 一个整数，指定处理问题时的批量大小。批量处理可以提高处理效率。
        tensorizer: 一个 Tensorizer 类的实例，用于将文本转换为模型可以处理的张量格式。
        """
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.selector = None

    def generate_question_vectors(self, questions: List[str], query_token: str = None) -> T:
        """ 
        将一组文本问题转换成向量
        questions: 一个字符串列表，包含需要转换的问题。
        query_token: （可选）一个字符串，表示查询时可能使用的特定令牌。它有一个默认值 None，表示不使用特定令牌。
        """

        bsz = self.batch_size   # 将批量大小赋值给局部变量 bsz
        self.question_encoder.eval()    # 将问题编码器设置为评估模式，这意味着在接下来的操作中，模型将不会进行任何训练相关的更改，如更新权重
        return generate_question_vectors(
            self.question_encoder,
            self.tensorizer,
            questions,
            bsz,
            query_token=query_token,
            selector=self.selector,
        )   # 负责实际的问题向量生成工作


class LocalFaissRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        # index 参数（类型为 DenseIndexer），用于存储和检索编码后的段落
        super().__init__(question_encoder, batch_size, tensorizer)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        path_id_prefixes: List = None,
    ):
        """
        从一组文件中读取编码后的段落，并将它们索引到 index 中。参数 vector_files 是包含向量数据文件路径的字符串列表，buffer_size 指定了一次性发送给索引进行索引的段落数量，path_id_prefixes 可用于指定路径ID前缀
        """
        buffer = []
        # 迭代编码文件，将读取的数据暂存于 buffer 中，直到其大小达到 buffer_size 后，就调用 index 的 index_data 方法对这些数据进行索引
        for i, item in enumerate(iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        # 对剩余在 buffer 中的数据进行索引，并记录索引完成的日志
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        接收查询向量和要检索的顶部文档数量（top_docs），返回与查询向量最匹配的顶部文档的列表
        首先记录开始时间，然后使用 index 的 search_knn 方法执行K近邻搜索，找到最匹配的 top_docs 个文档。记录搜索所花费的时间，并将 index 设置为 None（这可能是为了释放内存）。最后，返回搜索结果
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        self.index = None
        return results


# works only with our distributed_faiss library
class DenseRPCRetriever(DenseRetriever):
    # 通过分布式FAISS库实现密集向量检索
    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index_cfg_path: str,
        dim: int,
        use_l2_conversion: bool = False,
        nprobe: int = 256,
    ):
        """ 
        定义了类的初始化过程，包含了多个参数，如问题编码器(question_encoder)，批量大小(batch_size)，张量化器(tensorizer)，索引配置路径(index_cfg_path)，维度(dim)，是否使用L2转换(use_l2_conversion)，和探针数(nprobe)
        """
        from distributed_faiss.client import IndexClient # 用于与索引服务器进行通信

        super().__init__(question_encoder, batch_size, tensorizer)
        self.dim = dim
        self.index_id = "dr"    # 初始化索引ID为dr
        self.nprobe = nprobe
        logger.info("Connecting to index server ...")
        self.index_client = IndexClient(index_cfg_path) # 使用索引配置路径初始化IndexClient实例
        self.use_l2_conversion = use_l2_conversion
        logger.info("Connected")

    def load_index(self, index_id):
        # 加载远程索引
        from distributed_faiss.index_cfg import IndexCfg # 配置索引参数

        self.index_id = index_id
        logger.info("Loading remote index %s", index_id)
        idx_cfg = IndexCfg()
        idx_cfg.nprobe = self.nprobe # 设置探针数
        if self.use_l2_conversion:
            # 如果使用L2转换，则设置度量为L2
            idx_cfg.metric = "l2"

        # 通过IndexClient加载索引配置
        self.index_client.load_index(self.index_id, cfg=idx_cfg, force_reload=False)
        logger.info("Index loaded")
        # 等待索引准备就绪
        self._wait_index_ready(index_id)

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int = 1000,
        path_id_prefixes: List = None,
    ):
        """
        定义了一个方法用于将编码后的数据索引化
        """
        from distributed_faiss.index_cfg import IndexCfg

        buffer = [] # 定义一个缓冲区列表
        idx_cfg = IndexCfg()

        idx_cfg.dim = self.dim
        logger.info("Index train num=%d", idx_cfg.train_num)
        idx_cfg.faiss_factory = "flat"
        index_id = self.index_id
        self.index_client.create_index(index_id, idx_cfg)   # 创建索引

        def send_buf_data(buf, index_client):
            # 定义一个内部函数用于发送缓冲数据
            buffer_vectors = [np.reshape(encoded_item[1], (1, -1)) for encoded_item in buf]
            buffer_vectors = np.concatenate(buffer_vectors, axis=0)
            meta = [encoded_item[0] for encoded_item in buf]
            index_client.add_index_data(index_id, buffer_vectors, meta)

        for i, item in enumerate(iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)):
            # 遍历编码文件
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                send_buf_data(buffer, self.index_client) # 发送缓冲区数据进行索引
                buffer = []
        if buffer:
            send_buf_data(buffer, self.index_client) # 发送缓冲区数据进行索引
        logger.info("Embeddings sent.")
        self._wait_index_ready(index_id)    # 等待索引准备就绪

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100, search_batch: int = 512
    ) -> List[Tuple[List[object], List[float]]]:
        """
        基于查询向量批次检索最匹配文档的功能
        query_vectors: 查询向量的数组，每个向量代表一个查询。
        top_docs: 返回每个查询向量的顶部匹配文档数量，默认为100。
        search_batch: 查询处理的批次大小，默认为512。此参数控制一次处理的查询数量，有助于管理大规模查询的内存使用和性能。
        """
        if self.use_l2_conversion:
            # 如果启用了use_l2_conversion标志，该方法会添加一个额外的维度到查询向量中，并将其值设置为0。这主要用于在使用HNSW索引时兼容L2度量，从而保持查询向量和索引中的向量在同一空间中
            aux_dim = np.zeros(len(query_vectors), dtype="float32")
            query_vectors = np.hstack((query_vectors, aux_dim.reshape(-1, 1)))
            logger.info("query_hnsw_vectors %s", query_vectors.shape)
            self.index_client.cfg.metric = "l2"

        results = []
        # 遍历查询向量数组，每次处理一个批次的查询向量，执行检索操作
        for i in range(0, query_vectors.shape[0], search_batch):
            time0 = time.time()
            query_batch = query_vectors[i : i + search_batch]   # 获取当前批次的查询向量
            logger.info("query_batch: %s", query_batch.shape)
            # scores, meta = self.index_client.search(query_batch, top_docs, self.index_id)

            # 通过search_with_filter方法执行过滤搜索，它返回每个查询的顶部文档的分数和元数据。这里的过滤参数filter_pos和filter_value用于进一步细化搜索结果
            scores, meta = self.index_client.search_with_filter(
                query_batch, top_docs, self.index_id, filter_pos=3, filter_value=True
            )

            logger.info("index search time: %f sec.", time.time() - time0)
            # 将检索到的每个查询的顶部文档的分数和元数据组合为元组，然后添加到结果列表中
            results.extend([(meta[q], scores[q]) for q in range(len(scores))])
        # 方法返回一个列表，其中每个元素是一个元组。每个元组包含两个列表：匹配文档的元数据和相应的分数列表
        return results

    def _wait_index_ready(self, index_id: str):
        # 等待远程索引准备就绪, 参数 index_id 即索引的ID。
        # 从distributed_faiss.index_state模块导入IndexState枚举。IndexState提供了不同的索引状态，如TRAINED表示索引已经训练完成并准备就绪
        from distributed_faiss.index_state import IndexState
        # TODO: move this method into IndexClient class
        while self.index_client.get_state(index_id) != IndexState.TRAINED:
            # 这行代码开启了一个循环，持续检查索引的状态。self.index_client.get_state(index_id)调用返回给定index_id的当前状态。如果状态不是TRAINED，表示索引还未准备就绪，循环继续
            logger.info("Remote Index is not ready ...")
            time.sleep(60)  # 减少对索引状态的频繁检查，给远程索引准备提供足够的时间
        logger.info(
            "Remote Index is ready. Index data size %d",
            self.index_client.get_ntotal(index_id), # 调用返回索引中的数据总量
        )


def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    """ 
    验证给定文本片段（passages）和相应答案（answers）之间的匹配情况
    该函数接收五个参数，并返回一个列表，其中包含布尔值，用于表示每个问题的答案是否与顶部 K 个文档中的某个文档匹配
    
    五个参数：passages（一个字典，包含待验证的文本片段），answers（一个列表，包含对应的答案列表），result_ctx_ids（一个列表，包含与每个问题相关的文档ID及其相关性得分的元组），workers_num（工作线程数量），以及match_type（匹配类型）
    """
    logger.info("validating passages. size=%d", len(passages)) # 正在验证的文本片段数量
    # 评估 answers 与 passages 的匹配情况
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    # 一个列表，包含每个答案在顶部 K 个文档中被找到的次数
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    # 将 top_k_hits 列表中的每个值除以 result_ctx_ids 的长度（即问题的总数），以计算每个答案在顶部 K 个文档中的平均命中率
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    # 一个布尔值列表的列表，表示每个问题的答案是否在顶部 K 个文档中的某个文档中被命中
    return match_stats.questions_doc_hits


def validate_from_meta(
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
    meta_compressed: bool,
) -> List[List[bool]]:
    """ 
    对一组文档的元数据进行验证，看看它们是否包含了给定问题的答案
    
    answers: 一个列表，其中每个元素也是一个列表，这个内部的列表包含字符串，代表答案。
    result_ctx_ids: 一个列表，包含元组。每个元组有两个元素：一个对象列表和一个浮点数列表，代表结果上下文的ID和它们的评分或者相关度。
    workers_num: 一个整数，表示执行任务的工作线程数目。
    match_type: 一个字符串，指定匹配类型，如“精确匹配”、“模糊匹配”等。
    meta_compressed: 一个布尔值，指示元数据是否被压缩。
    返回值：一个布尔值的列表，每个列表对应一个问题，里面的布尔值表示相应的文档是否包含问题的答案。
    """

    # 在匹配过程中会使用文档的标题。这个函数的作用是根据提供的参数计算匹配的统计信息
    match_stats = calculate_matches_from_meta(
        answers, result_ctx_ids, workers_num, match_type, use_title=True, meta_compressed=meta_compressed
    )
    top_k_hits = match_stats.top_k_hits # 在前 K 个文档中找到答案的次数

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits] # 计算每个文档被击中的比例
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    # 这是一个布尔值的列表，每个子列表对应一个问题，里面的布尔值表示相应的文档是否包含问题的答案
    return match_stats.questions_doc_hits


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
):
    """ 
    将一系列问答数据和相关的通道（passages）信息合并，并保存到一个指定的文件中
    
    passages: 字典，键是对象（通常是通道的ID），值是包含文本和标题的元组。
    questions: 字符串列表，包含所有问题。
    answers: 每个问题的答案列表，其中每个答案也是一个列表。
    top_passages_and_scores: 每个问题的顶级通道及其分数，每个元素是一个元组，包含通道的ID列表和对应的分数列表。
    per_question_hits: 对于每个问题，一个布尔值列表表示每个通道是否含有答案。
    out_file: 结果保存的文件路径。
    """
    merged_data = [] # 存储合并后的数据
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):   # 获取每个问题的索引 i 和内容 q
        # 对于每个问题，从相应的索引处获取答案、顶级通道及其分数、以及是否含有答案的信息
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        # 通过索引访问 passages 获取每个通道的文本和标题。
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        # 将通道ID、标题、文本、分数以及是否含有答案合并成一个字典，然后添加到问题的 ctxs 字段
        results_item = {
            "question": q,
            "answers": q_answers,
            "ctxs": [
                {
                    "id": results_and_scores[0][c],
                    "title": docs[c][1],
                    "text": docs[c][0],
                    "score": scores[c],
                    "has_answer": hits[c],
                }
                for c in range(ctxs_num)
            ],
        }

        # if questions_extra_attr and questions_extra:
        #    extra = questions_extra[i]
        #    results_item[questions_extra_attr] = extra

        merged_data.append(results_item)

    with open(out_file, "w") as writer:
        # 将 merged_data 转换为格式化的JSON字符串，并写入文件
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


# TODO: unify with save_results
def save_results_from_meta(
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
    rpc_meta_compressed: bool = False,
):
    """ 
    将从元数据中得到的搜索结果以及相关信息保存到一个文件中
    
    questions: 一个字符串列表，包含查询的问题。
    answers: 一个列表的列表，内层列表包含每个问题的答案字符串。
    top_passages_and_scores: 包含文档段落和对应得分的元组列表，其中每个元组包含两个列表：第一个列表包含对象（这里指的是文档段落），第二个列表包含浮点数（这里指的是得分）。
    per_question_hits: 布尔值的列表的列表，表示每个问题的每个文档是否包含答案。
    out_file: 字符串，指定保存结果的文件路径。
    rpc_meta_compressed: 布尔值，指示是否需要对文档的标题和文本进行解压缩处理。
    """
    merged_data = [] # 存储合并后的数据
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        # 对于每个问题，提取对应的答案、文档及得分、以及是否包含答案的标志
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [doc for doc in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        # 对于每个问题，创建一个包含问题文本、答案列表和文档上下文的字典对象。文档上下文是一个字典列表，每个字典包含文档的ID、标题、文本、是否为维基、得分以及是否包含答案的标志。
        results_item = {
            "question": q,
            "answers": q_answers,
            "ctxs": [
                {
                    "id": docs[c][0],
                    # 如果 rpc_meta_compressed 为真，则使用 zlib.decompress 对文档的标题和文本进行解压缩并解码，否则直接使用
                    "title": zlib.decompress(docs[c][2]).decode() if rpc_meta_compressed else docs[c][2],
                    "text": zlib.decompress(docs[c][1]).decode() if rpc_meta_compressed else docs[c][1],
                    "is_wiki": docs[c][3],
                    "score": scores[c],
                    "has_answer": hits[c],
                }
                for c in range(ctxs_num)
            ],
        }
        # 将每个问题的结果添加到 merged_data 列表中
        merged_data.append(results_item)

    with open(out_file, "w") as writer:
        # 将 merged_data 列表转换为JSON格式并写入文件。这里使用 json.dumps 函数进行转换，并设置缩进为4，以便于阅读
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def iterate_encoded_files(vector_files: list, path_id_prefixes: List = None) -> Iterator[Tuple]:
    """ 
    迭代处理一系列编码后的文件，并生成处理后的文档向量数据
    
    vector_files: 一个列表，包含了需要处理的文件路径。
    path_id_prefixes: 一个可选的列表参数，默认为 None。如果提供，这个列表包含了与 vector_files 中每个文件对应的前缀字符串。这些前缀用于修改或补全文件中每个文档向量的标识符（ID）。
    """
    for i, file in enumerate(vector_files):
        # 同时返回文件的索引 (i) 和文件路径 (file)
        logger.info("Reading file %s", file)
        # 如果 path_id_prefixes 被提供，根据当前文件的索引 (i) 从中获取相应的ID前缀 (id_prefix)。否则，id_prefix 保持为 None。
        id_prefix = None
        if path_id_prefixes:
            id_prefix = path_id_prefixes[i]
        # 使用 open 函数以二进制读模式 ("rb") 打开当前文件 (file)，并将文件内容反序列化（即从二进制格式转换回Python对象），这里使用 pickle.load 来加载整个文档向量列表 (doc_vectors)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors: # 遍历 doc_vectors 中的每个文档向量 (doc)
                doc = list(doc) # 将 doc 转换成列表，以便可以修改它
                if id_prefix and not str(doc[0]).startswith(id_prefix):
                    # 如果提供了ID前缀 (id_prefix) 并且当前文档的ID (doc[0]) 不以这个前缀开头，则将这个前缀添加到文档ID的前面
                    doc[0] = id_prefix + str(doc[0])
                # 使用 yield 关键字返回处理后的文档向量。这使得 iterate_encoded_files 成为一个生成器函数，允许逐个处理文件中的文档向量而不是一次性加载整个文件内容到内存中。
                yield doc


def validate_tables(
    passages: Dict[object, TableChunk],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    """ 
    验证一组表格数据是否包含给定问题的答案，并计算其精确度
    
    passages: 字典，键是对象（通常指的是表格的ID），值是 TableChunk，一个包含表格文本、标题和ID的结构。
    answers: 每个问题的答案列表，其中每个答案也是一个列表。
    result_ctx_ids: 每个问题最接近的文档（或表格）ID和对应的分数，每个元素是一个元组，包含对象ID列表和分数列表。
    workers_num: 工作进程的数量，用于并行处理数据。
    match_type: 匹配类型，指定是进行精确匹配还是模糊匹配等。
    返回值：一个布尔值列表，每个列表对应一个问题，里面的布尔值表示相应的文档或表格块是否包含问题的答案。
    """ 
    # 计算给定的答案与表格数据块的匹配情况
    match_stats = calculate_chunked_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    # 提取匹配统计中的关键数据，包括 top_k_chunk_hits（表格块的命中数）和 top_k_table_hits（表格的命中数），并计算对应的精确度
    top_k_chunk_hits = match_stats.top_k_chunk_hits
    top_k_table_hits = match_stats.top_k_table_hits

    logger.info("Validation results: top k documents hits %s", top_k_chunk_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_chunk_hits]
    logger.info("Validation results: top k table chunk hits accuracy %s", top_k_hits)

    logger.info("Validation results: top k tables hits %s", top_k_table_hits)
    top_k_table_hits = [v / len(result_ctx_ids) for v in top_k_table_hits]
    logger.info("Validation results: top k tables accuracy %s", top_k_table_hits)

    # 返回 match_stats.top_k_chunk_hits，这个列表表示在前K个文档或表格块中找到答案的次数
    return match_stats.top_k_chunk_hits


def get_all_passages(ctx_sources):
    """ 
    从多个上下文源（ctx_sources）中加载和汇总通道数据（passages）。这里的“通道”可能指文档、文章段落或其他文本数据形式，常用于信息检索、问答系统等场景
    
    ctx_sources: 输入参数，通常是一个列表，包含多个上下文源对象。这些对象预计有一个方法 load_data_to，用于将其数据加载到指定的字典中。
    """
    all_passages = {} # 存储从所有上下文源中加载的通道数据
    for ctx_src in ctx_sources:
        # 调用每个上下文源的 load_data_to 方法，将其通道数据加载到 all_passages 字典中。这假设 load_data_to 方法以某种方式更新 all_passages 字典，可能是添加新的键值对或更新现有的
        ctx_src.load_data_to(all_passages)
        logger.info("Loaded ctx data: %d", len(all_passages)) # 记录到目前为止加载的通道数据数量

    if len(all_passages) == 0:
        # 如果是空的，说明没有成功加载任何通道数据，函数抛出一个 RuntimeError 异常，提示没有找到通道数据，并建议检查 ctx_file 参数是否正确指定
        raise RuntimeError("No passages data found. Please specify ctx_file param properly.")
    return all_passages

# 装饰器，用于指定Hydra配置的路径和名称。这个装饰器告诉Hydra从conf目录下加载名为dense_retriever的配置文件，然后将配置文件的内容作为参数cfg传递给main函数
@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg) # 根据配置初始化GPU设置，确保模型可以在GPU上运行以加速计算
    saved_state = load_states_from_checkpoint(cfg.model_file) # 从指定的检查点文件中加载模型状态

    set_cfg_params_from_state(saved_state.encoder_params, cfg) # 根据加载的模型状态调整配置参数

    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    # 初始化双向编码器的组件，这里主要是为了将文本转换成向量以及初始化编码模型
    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    logger.info("Loading saved model state ...")
    encoder.load_state(saved_state, strict=False)

    encoder_path = cfg.encoder_path
    if encoder_path:
        logger.info("Selecting encoder: %s", encoder_path)
        encoder = getattr(encoder, encoder_path)
    else:
        logger.info("Selecting standard question encoder")
        encoder = encoder.question_model

    # 设置编码器为分布式模式（如果适用）并将编码器设置为评估模式
    encoder, _ = setup_for_distributed_mode(encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)
    encoder.eval()

    # 确定编码器输出向量的大小，这对于之后的向量检索是必要的
    model_to_load = get_model_obj(encoder)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    # get questions & answers
    questions = []
    questions_text = []
    question_answers = []

    if not cfg.qa_dataset:
        logger.warning("Please specify qa_dataset to use")
        return

    ds_key = cfg.qa_dataset
    logger.info("qa_dataset: %s", ds_key)

    # 加载指定的问答数据集，并通过循环将问题和答案分别存储在列表中
    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()

    total_queries = len(qa_src)
    for i in range(total_queries):
        qa_sample = qa_src[i]
        question, answers = qa_sample.query, qa_sample.answers
        questions.append(question)
        question_answers.append(answers)

    logger.info("questions len %d", len(questions))
    logger.info("questions_text len %d", len(questions_text))

    # 根据配置，选择使用DenseRPCRetriever或LocalFaissRetriever作为检索器。这两种检索器分别用于远程过程调用(RPC)和本地检索
    if cfg.rpc_retriever_cfg_file:
        index_buffer_sz = 1000
        retriever = DenseRPCRetriever(
            encoder,
            cfg.batch_size,
            tensorizer,
            cfg.rpc_retriever_cfg_file,
            vector_size,
            use_l2_conversion=cfg.use_l2_conversion,
        )
    else:
        index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
        logger.info("Local Index class %s ", type(index))
        index_buffer_sz = index.buffer_size
        index.init_index(vector_size)
        retriever = LocalFaissRetriever(encoder, cfg.batch_size, tensorizer, index)

    logger.info("Using special token %s", qa_src.special_query_token)
    # 通过检索器为每个问题生成向量表示
    # 如果问答源（qa_src）指定了特殊的查询标记（special_query_token），则在生成向量时使用这个标记
    questions_tensor = retriever.generate_question_vectors(questions, query_token=qa_src.special_query_token)

    # 如果问答源配置了自定义的表示选择器（selector），则会使用这个选择器。表示选择器可以根据需要对编码器的输出进行调整或选择，以优化检索效果。
    if qa_src.selector:
        logger.info("Using custom representation token selector")
        retriever.selector = qa_src.selector

    index_path = cfg.index_path
    # 检查是否提供了RPC（远程过程调用）配置和索引ID，如果是，则通过RPC加载索引
    if cfg.rpc_retriever_cfg_file and cfg.rpc_index_id:
        retriever.load_index(cfg.rpc_index_id)
    # 如果指定了本地索引路径（index_path）且索引存在，则直接反序列化加载该索引
    elif index_path and index.index_exists(index_path):
        logger.info("Index path: %s", index_path)
        retriever.index.deserialize(index_path)
    else:
        # 存储上下文源的ID前缀和上下文源对象
        id_prefixes = []
        ctx_sources = []
        for ctx_src in cfg.ctx_datatsets: # 遍历配置中的上下文数据集
            # 根据配置实例化上下文源对象。Hydra允许通过配置动态创建对象，这使得代码更加灵活和可配置
            ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
            id_prefixes.append(ctx_src.id_prefix) # 上下文源的ID前缀
            ctx_sources.append(ctx_src) # 上下文源对象
            logger.info("ctx_sources: %s", type(ctx_src)) # 记录上下文源的类型信息

        # 记录每个数据集的ID前缀，这对于后续索引构建和检索是必要的，因为它允许系统区分来自不同上下文源的数据
        logger.info("id_prefixes per dataset: %s", id_prefixes)

        # 包含了编码后的上下文文件的路径模式。这些路径模式用于定位包含上下文信息（如文章或段落）的文件，这些信息将被编码并用于构建索引
        ctx_files_patterns = cfg.encoded_ctx_files

        logger.info("ctx_files_patterns: %s", ctx_files_patterns)
        if ctx_files_patterns:
            assert len(ctx_files_patterns) == len(id_prefixes), "ctx len={} pref leb={}".format(
                len(ctx_files_patterns), len(id_prefixes)
            )
        else:
            # 确保上下文文件模式的数量与ID前缀的数量相匹配。如果没有提供上下文文件模式，则要求指定索引路径(index_path)或RPC索引ID(cfg.rpc_index_id)，以确保索引可以被正确加载或创建
            assert (
                index_path or cfg.rpc_index_id
            ), "Either encoded_ctx_files or index_path pr rpc_index_id parameter should be set."

        input_paths = []
        path_id_prefixes = []
        for i, pattern in enumerate(ctx_files_patterns):
            # 使用glob.glob查找匹配该模式的所有文件。这允许动态地根据文件名模式选择文件，而不是硬编码文件路径
            pattern_files = glob.glob(pattern)
            pattern_id_prefix = id_prefixes[i]
            # 将找到的文件路径添加到input_paths列表中
            input_paths.extend(pattern_files)
            # 将对应的ID前缀添加到path_id_prefixes列表中，确保每个文件与其来源的ID前缀关联
            path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))
        # 记录将要索引的文件和对应的ID前缀
        logger.info("Embeddings files id prefixes: %s", path_id_prefixes)
        logger.info("Reading all passages data from files: %s", input_paths)
        # 将找到的文件和对应的ID前缀传入，以构建索引。这个过程涉及读取文件中的编码后的数据（如向量表示），并将它们添加到索引中
        retriever.index_encoded_data(input_paths, index_buffer_sz, path_id_prefixes=path_id_prefixes)
        # 如果提供了索引路径(index_path)，则在索引构建完成后，将索引序列化并保存到该路径。这允许将构建好的索引持久化，以便将来重新加载和使用，而无需重新执行索引构建过程
        if index_path:
            retriever.index.serialize(index_path)

    # get top k results
    # 根据问题的向量表示检索最相关的文档。这是通过比较问题向量与索引中存储的文档向量来实现的
    top_results_and_scores = retriever.get_top_docs(questions_tensor.numpy(), cfg.n_docs)

    # 根据配置，可能会使用不同的方法来验证检索到的文档是否满足预期的质量标准。这可以包括通过元数据验证、表格验证或其他自定义验证方法
    if cfg.use_rpc_meta:
        questions_doc_hits = validate_from_meta(
            question_answers,
            top_results_and_scores,
            cfg.validation_workers,
            cfg.match,
            cfg.rpc_meta_compressed,
        )
        if cfg.out_file:
            save_results_from_meta(
                questions,
                question_answers,
                top_results_and_scores,
                questions_doc_hits,
                cfg.out_file,
                cfg.rpc_meta_compressed,
            )
    else:
        all_passages = get_all_passages(ctx_sources)
        if cfg.validate_as_tables:

            questions_doc_hits = validate_tables(
                all_passages,
                question_answers,
                top_results_and_scores,
                cfg.validation_workers,
                cfg.match,
            )

        else:
            questions_doc_hits = validate(
                all_passages,
                question_answers,
                top_results_and_scores,
                cfg.validation_workers,
                cfg.match,
            )

        # 如果配置了输出文件（out_file），则将检索和验证后的结果保存到该文件中
        if cfg.out_file:
            save_results(
                all_passages,
                questions_text if questions_text else questions,
                question_answers,
                top_results_and_scores,
                questions_doc_hits,
                cfg.out_file,
            )

    # 如果指定了kilt_out_file，则将结果转换为KILT格式。KILT是一种标准格式，用于知识强化的文本生成任务，支持结果的标准化评估和比较
    if cfg.kilt_out_file:
        kilt_ctx = next(iter([ctx for ctx in ctx_sources if isinstance(ctx, KiltCsvCtxSrc)]), None)
        if not kilt_ctx:
            raise RuntimeError("No Kilt compatible context file provided")
        assert hasattr(cfg, "kilt_out_file")
        kilt_ctx.convert_to_kilt(qa_src.kilt_gold_file, cfg.out_file, cfg.kilt_out_file)


if __name__ == "__main__":
    main()
