import glob
import logging
import os

import hydra
from omegaconf import DictConfig

from dpr.data.biencoder_data import JsonQADataset

logger = logging.getLogger(__name__)


class BiencoderDatasetsCfg(object): # 初始化双编码器模型数据集配置的类
    def __init__(self, cfg: DictConfig): # 根据提供的配置(DictConfig)初始化训练和验证(开发)数据集
        ds_cfg = cfg.datasets # 从总配置中提取数据集的配置部分
        self.train_datasets_names = cfg.train_datasets # 从配置中获取训练数据集的名称列表
        logger.info("train_datasets: %s", self.train_datasets_names) # 记录训练数据集的名称
        self.train_datasets = _init_datasets(self.train_datasets_names, ds_cfg)
        self.dev_datasets_names = cfg.dev_datasets # 从配置中获取验证数据集的名称列表
        logger.info("dev_datasets: %s", self.dev_datasets_names) # 记录验证数据集的名称
        self.dev_datasets = _init_datasets(self.dev_datasets_names, ds_cfg)
        self.sampling_rates = cfg.train_sampling_rates # 从配置中获取训练数据集的采样率


def _init_datasets(datasets_names, ds_cfg: DictConfig): 
    # 接收数据集名称（单个字符串或字符串列表）和数据集配置对象ds_cfg
    if isinstance(datasets_names, str):
        # 如果datasets_names是一个字符串，表示只有一个数据集，那么直接使用_init_dataset函数初始化这个数据集并返回包含单个数据集的列表
        return [_init_dataset(datasets_names, ds_cfg)]
    elif datasets_names:
        return [_init_dataset(ds_name, ds_cfg) for ds_name in datasets_names]
    else:
        return []


def _init_dataset(name: str, ds_cfg: DictConfig):
    if os.path.exists(name):
        # use default biencoder json class
        # 首先检查名称是否对应一个存在的路径。如果是，假设数据集是一个JSON格式的问答数据集，使用JsonQADataset类初始化并返回这个数据集
        return JsonQADataset(name)
    elif glob.glob(name):
        # 如果名称对应的路径不存在，尝试使用glob.glob查找匹配的文件。如果找到，对每个找到的文件递归调用_init_dataset
        files = glob.glob(name)
        return [_init_dataset(f, ds_cfg) for f in files]
    # try to find in cfg
    if name not in ds_cfg:
        # 如果既不是路径也不在配置中定义，抛出一个运行时错误，指明无法找到相应的数据集配置或位置
        raise RuntimeError("Can't find dataset location/config for: {}".format(name))
    return hydra.utils.instantiate(ds_cfg[name]) # 如果名称既不是存在的路径也没有匹配的glob模式，尝试在配置中查找相应的配置。如果找到，使用Hydra的instantiate方法根据配置实例化数据集
