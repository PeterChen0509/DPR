import collections
import csv
import json
import logging
import pickle
from typing import Dict, List

import hydra
import jsonlines
import torch
from omegaconf import DictConfig

from dpr.data.biencoder_data import (
    BiEncoderPassage,
    normalize_passage,
    get_dpr_files,
    read_nq_tables_jsonl,
    split_tables_to_chunks,
)

from dpr.utils.data_utils import normalize_question

logger = logging.getLogger(__name__)

TableChunk = collections.namedtuple("TableChunk", ["text", "title", "table_id"]) # 命名元组


class QASample:
    def __init__(self, query: str, id, answers: List[str]):
        self.query = query
        self.id = id
        self.answers = answers


class RetrieverData(torch.utils.data.Dataset):
    # 加载特定的数据文件
    # torch.utils.data.Dataset是PyTorch中用于处理数据集的一个基础类，通过继承它，可以自定义数据加载方式，以适应各种不同的数据处理需求
    def __init__(self, file: str):
        """
        :param file: - real file name or the resource name as they are defined in download_data.py
        """
        self.file = file
        self.data_files = [] # 存储通过file参数指定的数据文件的路径

    def load_data(self):
        # 根据提供的文件名或资源名找到相应的数据文件，并返回这些文件的路径列表
        self.data_files = get_dpr_files(self.file)
        # 确保self.data_files列表中只包含一个文件路径。如果包含多个文件路径，将抛出一个异常，并提示当前只支持单个文件
        assert (
            len(self.data_files) == 1
        ), "RetrieverData source currently works with single files only. Files specified: {}".format(self.data_files)
        # 如果self.data_files列表中确实只有一个元素，这行代码将这个元素（即数据文件的路径）赋值给self.file属性。这意味着之后可以直接使用self.file来访问数据文件
        self.file = self.data_files[0]


class QASrc(RetrieverData):
    # QASrc类被设计来处理特定于问答（QA）场景的数据，提供了加载和预处理问答数据集的功能
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file) # 调用父类RetrieverData的构造函数，传递file参数
        self.data = None # 存储加载的问答数据
        # 如果selector参数被提供，则通过Hydra框架的instantiate方法实例化该配置对象
        self.selector = hydra.utils.instantiate(selector) if selector else None
        # 处理查询时添加特定的标记或后缀
        self.special_query_token = special_query_token
        self.query_special_suffix = query_special_suffix

    def __getitem__(self, index) -> QASample:
        # 如何通过索引访问数据集中的元素。这个方法返回self.data[index]，即数据列表中的第index个问答样本
        return self.data[index]

    def __len__(self):
        # 数据集的大小
        return len(self.data)

    def _process_question(self, question: str): # 预处理问题文本
        # as of now, always normalize query
        question = normalize_question(question) # 去双引号 -> 单引号
        # 如果定义了查询的特殊后缀且当前问题文本不以该后缀结尾，则将该后缀添加到问题文本的末尾
        if self.query_special_suffix and not question.endswith(self.query_special_suffix):
            question += self.query_special_suffix
        return question


class CsvQASrc(QASrc):
    def __init__(
        self,
        file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
        data_range_start: int = -1,
        data_size: int = -1,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_col = question_col
        self.answers_col = answers_col
        self.id_col = id_col
        self.data_range_start = data_range_start
        self.data_size = data_size

    def load_data(self):
        super().load_data()
        data = []
        start = self.data_range_start
        # size = self.data_size
        samples_count = 0
        # TODO: optimize
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                question = row[self.question_col]
                answers = eval(row[self.answers_col])
                id = None
                if self.id_col >= 0:
                    id = row[self.id_col]
                samples_count += 1
                # if start !=-1 and samples_count<=start:
                #    continue
                data.append(QASample(self._process_question(question), id, answers))

        if start != -1:
            end = start + self.data_size if self.data_size != -1 else -1
            logger.info("Selecting dataset range [%s,%s]", start, end)
            self.data = data[start:end] if end != -1 else data[start:]
        else:
            self.data = data


class JsonlQASrc(QASrc):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        question_attr: str = "question",
        answers_attr: str = "answers",
        id_attr: str = "id",
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_attr = question_attr
        self.answers_attr = answers_attr
        self.id_attr = id_attr

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                answers = jline[self.answers_attr] if self.answers_attr in jline else []
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class KiltCsvQASrc(CsvQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
        data_range_start: int = -1,
        data_size: int = -1,
    ):
        super().__init__(
            file,
            question_col,
            answers_col,
            id_col,
            selector,
            special_query_token,
            query_special_suffix,
            data_range_start,
            data_size,
        )
        self.kilt_gold_file = kilt_gold_file


class KiltJsonlQASrc(JsonlQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_attr: str = "input",
        answers_attr: str = "answer",
        id_attr: str = "id",
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(
            file,
            selector,
            question_attr,
            answers_attr,
            id_attr,
            special_query_token,
            query_special_suffix,
        )
        self.kilt_gold_file = kilt_gold_file

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                out = jline["output"]
                answers = [o["answer"] for o in out if "answer" in o]
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class TTS_ASR_QASrc(QASrc):
    def __init__(self, file: str, trans_file: str):
        super().__init__(file)
        self.trans_file = trans_file

    def load_data(self):
        super().load_data()
        orig_data_dict = {}
        with open(self.file, "r") as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            id = 0
            for row in reader:
                question = row[0]
                answers = eval(row[1])
                orig_data_dict[id] = (question, answers)
                id += 1
        data = []
        with open(self.trans_file, "r") as tfile:
            reader = csv.reader(tfile, delimiter="\t")
            for r in reader:
                row_str = r[0]
                idx = row_str.index("(None-")
                q_id = int(row_str[idx + len("(None-") : -1])
                orig_data = orig_data_dict[q_id]
                answers = orig_data[1]
                q = row_str[:idx].strip().lower()
                data.append(QASample(q, idx, answers))
        self.data = data


class CsvCtxSrc(RetrieverData):
    # 从CSV文件中加载并处理文本数据，以便用于信息检索或其他需要编码器（如双编码器）的NLP任务
    def __init__(
        self,
        file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        """ 
        file: CSV文件的路径。
        id_col: 唯一标识符所在列的索引。
        text_col: 存储文本内容（如段落或文章）的列索引。
        title_col: 存储标题的列索引。
        id_prefix: 可选参数，指定一个前缀，该前缀将添加到每个条目的ID前面，用于生成唯一标识符。
        normalize: 布尔值，指示是否对文本内容进行标准化处理。
        """
        super().__init__(file)
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        # 读取CSV文件，并将数据加载到传入的字典ctxs中，该字典的键为文本的唯一标识符，值为BiEncoderPassage对象
        super().load_data() # 调用父类的load_data方法，可能进行一些基本的数据加载或预处理
        logger.info("Reading file %s", self.file)
        with open(self.file) as ifile:
            # 使用csv.reader读取CSV文件。文件使用制表符\t作为分隔符
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:  # 遍历文件中的每一行
                # for row in ifile:
                # row = row.strip().split("\t")
                # 如果行的ID列值为"id"，则跳过该行。这可能用于跳过文件头
                if row[self.id_col] == "id":
                    continue
                # 根据是否指定了id_prefix，构建每个条目的唯一标识符
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[self.id_col])
                else:
                    sample_id = row[self.id_col]
                passage = row[self.text_col].strip('"')
                # 如果启用了normalize选项，则调用normalize_passage函数对文本进行标准化处理
                if self.normalize:
                    passage = normalize_passage(passage)
                # 创建BiEncoderPassage对象，其中包含文本和标题，并将其与相应的唯一标识符关联在ctxs字典中
                ctxs[sample_id] = BiEncoderPassage(passage, row[self.title_col])


class KiltCsvCtxSrc(CsvCtxSrc):
    # 处理CSV格式的上下文源（如文章、段落等）并将它们转换为KILT（Knowledge Intensive Language Tasks）格式，以便与KILT兼容的任务和数据集使用
    def __init__(
        self,
        file: str,
        mapping_file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        """ 
        文件路径file，映射文件路径mapping_file，以及一些可选参数，包括ID列id_col，文本列text_col，标题列title_col，ID前缀id_prefix，以及是否正规化文本的标志normalize
        """
        # 调用父类CsvCtxSrc的构造函数，并传递相应的参数
        super().__init__(file, id_col, text_col, title_col, id_prefix, normalize=normalize)
        # 将传入的映射文件路径赋值给实例变量
        # mapping_file。这个映射文件通常包含从DPR（Dense Passage Retrieval）ID到KILT格式所需信息（如Wikipedia ID）的映射
        self.mapping_file = mapping_file

    def convert_to_kilt(self, kilt_gold_file, dpr_output, kilt_out_file):
        # 定义了一个方法，用于将DPR输出转换为KILT格式。该方法接受三个参数：KILT的黄金标准文件kilt_gold_file，DPR的输出文件dpr_output，以及转换后的KILT格式输出文件路径kilt_out_file
        logger.info("Converting to KILT format file: %s", dpr_output)

        # 使用json.load读取DPR输出文件，该文件包含了DPR模型的预测结果
        with open(dpr_output, "rt") as fin:
            dpr_output = json.load(fin)

        # 读取KILT黄金标准文件：使用jsonlines.open以只读模式打开KILT黄金标准文件，并将其内容转换为列表
        with jsonlines.open(kilt_gold_file, "r") as reader:
            kilt_gold_file = list(reader)
        # 断言两个文件的长度相等：确保DPR输出和KILT黄金标准文件中的条目数相同
        assert len(kilt_gold_file) == len(dpr_output)
        # 读取并反序列化映射文件，该文件包含从DPR ID到KILT所需信息的映射
        map_path = self.mapping_file
        with open(map_path, "rb") as fin:
            mapping = pickle.load(fin)

        with jsonlines.open(kilt_out_file, mode="w") as writer:
            # 以写入模式打开KILT输出文件，并逐个处理DPR输出和KILT黄金标准文件中的条目
            for dpr_entry, kilt_gold_entry in zip(dpr_output, kilt_gold_file):
                # assert dpr_entry["question"] == kilt_gold_entry["input"]
                provenance = [] # 创建一个空列表provenance用于存储来源信息
                for ctx in dpr_entry["ctxs"]:
                    # 遍历DPR条目中的上下文（ctxs），使用映射文件转换每个上下文的ID到所需的KILT格式信息，并添加到provenance列表中
                    wikipedia_id, end_paragraph_id = mapping[int(ctx["id"])]
                    provenance.append(
                        {
                            "wikipedia_id": wikipedia_id,
                            "end_paragraph_id": end_paragraph_id,
                        }
                    )
                # 构建KILT条目，包括ID、输入问题、输出（包括来源信息），然后写入到KILT输出文件
                kilt_entry = {
                    "id": kilt_gold_entry["id"],
                    "input": kilt_gold_entry["input"],  # dpr_entry["question"],
                    "output": [{"provenance": provenance}],
                }
                writer.write(kilt_entry)

        # 记录日志信息，指出KILT格式的结果已保存到指定文件
        logger.info("Saved KILT formatted results to: %s", kilt_out_file)


class JsonlTablesCtxSrc(object):
    def __init__(
        self,
        file: str,
        tables_chunk_sz: int = 100,
        split_type: str = "type1",
        id_prefix: str = None,
    ):
        self.tables_chunk_sz = tables_chunk_sz
        self.split_type = split_type
        self.file = file
        self.id_prefix = id_prefix

    def load_data_to(self, ctxs: Dict):
        docs = {}
        logger.info("Parsing Tables data from: %s", self.file)
        tables_dict = read_nq_tables_jsonl(self.file)
        table_chunks = split_tables_to_chunks(tables_dict, self.tables_chunk_sz, split_type=self.split_type)
        for chunk in table_chunks:
            sample_id = self.id_prefix + str(chunk[0])
            docs[sample_id] = TableChunk(chunk[1], chunk[2], chunk[3])
        logger.info("Loaded %d tables chunks", len(docs))
        ctxs.update(docs)
