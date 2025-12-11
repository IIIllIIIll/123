#!/usr/bin/env python3
"""
公共模型组件
提取CFG和FCG模型中的共同组件，减少代码冗余
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger(__name__)


class BaseAttentionPooling(nn.Module):
    """基础注意力池化层 - CFG和FCG的通用实现"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        if num_heads == 1:
            # 单头注意力 (CFG风格)
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # 多头注意力 (FCG风格)
            self.head_dim = input_dim // num_heads
            assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
            
            self.query = nn.Linear(input_dim, input_dim)
            self.key = nn.Linear(input_dim, input_dim)
            self.value = nn.Linear(input_dim, input_dim)
            self.dropout = nn.Dropout(0.1)
            self.output_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x, batch):
        batch_size = batch.max().item() + 1
        pooled_features = []
        
        for i in range(batch_size):
            mask = (batch == i)
            graph_x = x[mask]
            
            if graph_x.size(0) == 0:
                pooled_features.append(torch.zeros(1, x.size(1), device=x.device))
                continue
            
            if self.num_heads == 1:
                # 单头注意力
                attn_weights = self.attention(graph_x)
                attn_weights = F.softmax(attn_weights, dim=0)
                weighted_x = (graph_x * attn_weights).sum(dim=0, keepdim=True)
                pooled_features.append(weighted_x)
            else:
                # 多头注意力
                q = self.query(graph_x).view(-1, self.num_heads, self.head_dim)
                k = self.key(graph_x).view(-1, self.num_heads, self.head_dim)
                v = self.value(graph_x).view(-1, self.num_heads, self.head_dim)
                
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)
                
                attended = torch.matmul(attn_weights, v)
                attended = attended.view(-1, x.size(1))
                
                graph_repr = torch.mean(attended, dim=0, keepdim=True)
                graph_repr = self.output_proj(graph_repr)
                pooled_features.append(graph_repr)
        
        return torch.cat(pooled_features, dim=0)


class BaseGraphEncoder(nn.Module):
    """基础图编码器 - CFG和FCG的通用实现"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 3, dropout: float = 0.5, use_residual: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_residual = use_residual
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GCN层
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            self.conv_layers.append(GCNConv(in_dim, out_dim))
        
        # 批归一化层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        # 输入投影
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GCN层
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            residual = x if self.use_residual else None
            
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout_layer(x)
            
            # 残差连接
            if self.use_residual and residual is not None and residual.size() == x.size():
                x = x + residual
        
        # 输出投影
        x = self.output_proj(x)
        
        return x


class BaseGraphClassifier(nn.Module):
    """基础图分类器 - CFG和FCG的通用实现"""
    
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, 
                 num_classes: int, num_layers: int = 3, dropout: float = 0.5,
                 pooling_type: str = 'attention', num_attention_heads: int = 1):
        super().__init__()
        
        # 图编码器
        self.encoder = BaseGraphEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_residual=True
        )
        
        # 池化层
        if pooling_type == 'attention':
            self.pooling = BaseAttentionPooling(
                embedding_dim, 
                hidden_dim // 2, 
                num_attention_heads
            )
            classifier_input_dim = embedding_dim
        else:
            # 使用全局池化
            self.pooling = None
            classifier_input_dim = embedding_dim * 2  # mean + max pooling
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def get_embedding(self, x, edge_index, batch):
        """
        [新增] 提取图嵌入向量
        """
        # 图编码
        node_embeddings = self.encoder(x, edge_index)
        
        # 图级池化
        if self.pooling is not None:
            graph_embeddings = self.pooling(node_embeddings, batch)
        else:
            # 使用全局池化
            mean_pool = global_mean_pool(node_embeddings, batch)
            max_pool = global_max_pool(node_embeddings, batch)
            graph_embeddings = torch.cat([mean_pool, max_pool], dim=1)
        
        return graph_embeddings

    def forward(self, x, edge_index, batch):
        """
        [修改] 使用 get_embedding
        """
        # 提取图嵌入
        graph_embeddings = self.get_embedding(x, edge_index, batch)
        
        # 分类
        logits = self.classifier(graph_embeddings)
        
        return logits


class BaseDataLoader:
    """基础数据加载器 - CFG和FCG的通用实现"""
    
    def __init__(self, data_dir: str, cache_size: int = 2000, enable_cache: bool = True, suffix: str = ".pt"):
        self.data_dir = Path(data_dir)
        self.cache_size = cache_size
        self.enable_cache = enable_cache
        self.suffix = suffix  # 新增：文件后缀参数
        
        # LRU缓存
        if self.enable_cache:
            self._cache = OrderedDict()
        
        # 扫描可用样本
        self._available_samples = {}
        self._scan_available_samples()
        
        logger.info(f"数据加载器初始化完成，数据目录: {data_dir}")
        logger.info(f"缓存设置: {'启用' if enable_cache else '禁用'}，缓存大小: {cache_size}")
        logger.info(f"文件后缀: {suffix}")
    
    def _scan_available_samples(self):
        """扫描可用的样本文件"""
        self._available_samples = {}
        
        if not self.data_dir.exists():
            logger.warning(f"数据目录不存在: {self.data_dir}")
            return
        
        # 扫描文件 (使用 glob 匹配后缀)
        pt_files = list(self.data_dir.glob(f"**/*{self.suffix}"))
        
        for pt_file in pt_files:
            # 关键修改：从文件名中移除后缀，得到 sample_id
            sample_id = pt_file.name.replace(self.suffix, '')
            self._available_samples[sample_id] = {
                'data_path': pt_file,
                'format': 'pt'
            }
        
        logger.info(f"扫描到 {len(self._available_samples)} 个样本 (后缀: {self.suffix})")
    
    def _load_sample(self, sample_id: str) -> Optional[Data]:
        """加载单个样本"""
        if sample_id not in self._available_samples:
            logger.warning(f"样本不存在: {sample_id}")
            return None
        
        # 检查缓存
        if self.enable_cache and sample_id in self._cache:
            # 移动到末尾（LRU）
            self._cache.move_to_end(sample_id)
            return self._cache[sample_id]
        
        # 加载数据
        sample_info = self._available_samples[sample_id]
        data_path = sample_info['data_path']
        
        try:
            data = torch.load(data_path, map_location='cpu')
            
            # 添加到缓存
            if self.enable_cache:
                if len(self._cache) >= self.cache_size:
                    # 移除最旧的项
                    self._cache.popitem(last=False)
                self._cache[sample_id] = data
            
            return data
            
        except Exception as e:
            logger.error(f"加载样本失败 {sample_id}: {e}")
            return None
    
    def get_available_samples(self) -> List[str]:
        """获取所有可用样本ID"""
        return list(self._available_samples.keys())
    
    def load_batch(self, sample_ids: List[str]) -> List[Data]:
        """批量加载样本"""
        batch_data = []
        
        for sample_id in sample_ids:
            data = self._load_sample(sample_id)
            if data is not None:
                batch_data.append(data)
        
        return batch_data


def create_model_from_config(config, model_type: str = 'classifier'):
    """
    根据配置创建模型
    
    Args:
        config: 配置对象
        model_type: 模型类型 ('classifier', 'encoder')
    
    Returns:
        模型实例
    """
    if model_type == 'classifier':
        # 从配置中获取池化参数，如果不存在则使用默认值
        pooling_type = getattr(config, 'POOLING_TYPE', 'attention')
        num_attention_heads = getattr(config, 'NUM_ATTENTION_HEADS', 1)
        
        return BaseGraphClassifier(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            embedding_dim=config.EMBEDDING_DIM,
            num_classes=config.NUM_CLASSES,
            num_layers=config.NUM_GCN_LAYERS,
            dropout=config.DROPOUT,
            pooling_type=pooling_type,
            num_attention_heads=num_attention_heads
        )
    elif model_type == 'encoder':
        return BaseGraphEncoder(
            input_dim=config.INPUT_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.EMBEDDING_DIM,
            num_layers=config.NUM_GCN_LAYERS,
            dropout=config.DROPOUT,
            use_residual=True
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")