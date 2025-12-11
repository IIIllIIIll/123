#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFG特征提取器 - 用于GCN输入构建

本模块处理CFG（控制流图）JSON文件，构建GCN输入数据。
它构建图结构（节点映射和edge_index关系矩阵），并使用100维FastText模型构建X特征矩阵。

主要功能：
1. 节点映射：将global_id映射到连续索引
2. 边索引构建：创建PyTorch Geometric兼容的edge_index
3. 特征矩阵构建：使用100维FastText模型处理操作码序列
4. 批量处理：高效处理多个CFG JSON文件


"""

import json
import os
import numpy as np
import torch
from torch_geometric.data import Data
import fasttext
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import pickle

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_opcode_sequence(opcode_sequence: List[str]) -> Optional[str]:
    """
    清理和处理操作码序列，用于FastText输入。
    
    参数：
        opcode_sequence: 来自CFG节点的操作码列表
        
    返回：
        清理后的操作码字符串用于FastText，如果无效则返回None
    """
    if not opcode_sequence or not isinstance(opcode_sequence, list):
        return None
    
    # 过滤掉空字符串和None值
    cleaned_opcodes = [op.strip() for op in opcode_sequence if op and isinstance(op, str) and op.strip()]
    
    if not cleaned_opcodes:
        return None
    
    # 将操作码列表连接成单个字符串，用空格分隔
    return ' '.join(cleaned_opcodes)


def build_node_mapping(cfg_data: Dict) -> Dict[str, int]:
    """
    构建从global_id到连续索引的节点映射。
    
    参数：
        cfg_data: CFG JSON数据
        
    返回：
        从global_id到索引的映射字典
    """
    global_nodes = cfg_data.get('global_nodes', [])
    
    # 提取所有global_id并创建映射
    global_ids = [node['global_id'] for node in global_nodes if 'global_id' in node]
    node_mapping = {global_id: idx for idx, global_id in enumerate(global_ids)}
    
    return node_mapping


def construct_edge_index(cfg_data: Dict, node_mapping: Dict[str, int]) -> torch.Tensor:
    """
    构建PyTorch Geometric兼容的edge_index张量。
    
    参数：
        cfg_data: CFG JSON数据
        node_mapping: global_id到索引的映射
        
    返回：
        形状为[2, num_edges]的edge_index张量
    """
    global_edges = cfg_data.get('global_edges', [])
    
    # 收集有效的边
    edges = []
    for edge in global_edges:
        source = edge.get('source')
        target = edge.get('target')
        
        # 检查源节点和目标节点是否在映射中
        if source in node_mapping and target in node_mapping:
            source_idx = node_mapping[source]
            target_idx = node_mapping[target]
            edges.append([source_idx, target_idx])
    
    if not edges:
        # 如果没有边，返回空的edge_index
        return torch.empty((2, 0), dtype=torch.long)
    
    # 转换为张量并转置以获得[2, num_edges]的形状
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return edge_index


def construct_feature_matrix(cfg_data: Dict, node_mapping: Dict[str, int], 
                           fasttext_model) -> torch.Tensor:
    """
    使用100维FastText模型构建特征矩阵。
    
    参数：
        cfg_data: CFG JSON数据
        node_mapping: global_id到索引的映射
        fasttext_model: 加载的FastText模型
        
    返回：
        形状为[num_nodes, 100]的特征矩阵
    """
    global_nodes = cfg_data.get('global_nodes', [])
    
    # 创建节点ID到节点数据的映射
    node_data_map = {node['global_id']: node for node in global_nodes if 'global_id' in node}
    
    # 初始化特征矩阵
    num_nodes = len(node_mapping)
    feature_matrix = torch.zeros((num_nodes, 100), dtype=torch.float32)
    
    # 为每个节点构建特征向量
    for global_id, idx in node_mapping.items():
        if global_id in node_data_map:
            node = node_data_map[global_id]
            opcode_sequence = node.get('opcode_sequence', [])
            
            # 清理操作码序列
            cleaned_opcodes = clean_opcode_sequence(opcode_sequence)
            
            if cleaned_opcodes:
                try:
                    # 使用FastText获取句子向量
                    feature_vector = fasttext_model.get_sentence_vector(cleaned_opcodes)
                    feature_matrix[idx] = torch.tensor(feature_vector, dtype=torch.float32)
                except Exception as e:
                    logger.warning(f"无法为节点 {global_id} 获取特征向量: {e}")
                    # 使用零向量作为默认值
                    feature_matrix[idx] = torch.zeros(100, dtype=torch.float32)
            else:
                # 如果没有有效的操作码，使用零向量
                feature_matrix[idx] = torch.zeros(100, dtype=torch.float32)
    
    return feature_matrix


def process_single_cfg(cfg_file_path: str, output_dir: str, fasttext_model) -> bool:
    """
    处理单个CFG JSON文件并保存GCN输入数据。
    
    参数：
        cfg_file_path: CFG JSON文件路径
        output_dir: 输出目录
        fasttext_model: 加载的FastText模型
        
    返回：
        处理成功返回True，否则返回False
    """
    try:
        # 加载CFG JSON数据
        with open(cfg_file_path, 'r', encoding='utf-8') as f:
            cfg_data = json.load(f)
        
        # 构建节点映射
        node_mapping = build_node_mapping(cfg_data)
        
        if not node_mapping:
            logger.warning(f"在 {cfg_file_path} 中未找到有效节点")
            return False
        
        # 构建边索引
        edge_index = construct_edge_index(cfg_data, node_mapping)
        
        # 构建特征矩阵
        x = construct_feature_matrix(cfg_data, node_mapping, fasttext_model)
        
        # 创建PyTorch Geometric数据对象
        graph_data = Data(x=x, edge_index=edge_index)
        
        # 添加元数据
        graph_data.num_nodes = len(node_mapping)
        graph_data.num_edges = edge_index.shape[1]
        
        # 保存处理后的数据
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(cfg_file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_gcn_data.pt")
        
        torch.save(graph_data, output_file)
        
        logger.info(f"已处理 {cfg_file_path} -> {output_file}")
        logger.info(f"节点数: {graph_data.num_nodes}, 边数: {graph_data.num_edges}")
        
        return True
        
    except Exception as e:
        logger.error(f"处理 {cfg_file_path} 时出错: {e}")
        return False


def batch_process_cfg(input_dir: str, output_dir: str, fasttext_model_path: str) -> Dict[str, int]:
    """
    批量处理CFG JSON文件。
    
    参数：
        input_dir: 包含CFG JSON文件的输入目录
        output_dir: 输出目录
        fasttext_model_path: FastText模型文件路径
        
    返回：
        包含处理统计信息的字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载FastText模型
    logger.info(f"正在加载FastText模型: {fasttext_model_path}")
    try:
        fasttext_model = fasttext.load_model(fasttext_model_path)
        logger.info("FastText模型加载成功")
    except Exception as e:
        logger.error(f"加载FastText模型失败: {e}")
        return {'successful': 0, 'failed': 0, 'total': 0}
    
    # 查找所有JSON文件
    json_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    logger.info(f"找到 {len(json_files)} 个JSON文件需要处理")
    
    # 处理文件
    successful_count = 0
    failed_count = 0
    
    for json_file in json_files:
        try:
            # 处理CFG文件
            success = process_single_cfg(json_file, output_dir, fasttext_model)
            
            if success:
                successful_count += 1
                
                if successful_count % 100 == 0:
                    logger.info(f"已处理 {successful_count} 个文件...")
            else:
                failed_count += 1
                
        except Exception as e:
            logger.error(f"处理 {json_file} 时出错: {e}")
            failed_count += 1
    
    logger.info(f"批量处理完成: {successful_count} 成功, {failed_count} 失败")
    
    return {
        'successful': successful_count,
        'failed': failed_count,
        'total': len(json_files)
    }


def analyze_cfg_data(input_dir: str) -> Dict[str, Any]:
    """
    分析CFG数据集的统计信息。
    
    参数：
        input_dir: 包含CFG JSON文件的输入目录
        
    返回：
        包含数据集统计信息的字典
    """
    try:
        stats = {
            'total_files': 0,
            'total_nodes': 0,
            'total_edges': 0,
            'avg_nodes_per_file': 0,
            'avg_edges_per_file': 0,
            'node_count_distribution': {},
            'edge_count_distribution': {}
        }
        
        # 查找所有JSON文件
        json_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        stats['total_files'] = len(json_files)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    cfg_data = json.load(f)
                
                # 统计节点和边数量
                nodes = cfg_data.get('global_nodes', [])
                edges = cfg_data.get('global_edges', [])
                
                node_count = len(nodes)
                edge_count = len(edges)
                
                stats['total_nodes'] += node_count
                stats['total_edges'] += edge_count
                
                # 更新分布统计
                stats['node_count_distribution'][node_count] = stats['node_count_distribution'].get(node_count, 0) + 1
                stats['edge_count_distribution'][edge_count] = stats['edge_count_distribution'].get(edge_count, 0) + 1
                
            except Exception as e:
                logger.warning(f"分析 {json_file} 时出错: {e}")
                continue
        
        # 计算平均值
        if stats['total_files'] > 0:
            stats['avg_nodes_per_file'] = stats['total_nodes'] / stats['total_files']
            stats['avg_edges_per_file'] = stats['total_edges'] / stats['total_files']
        
        logger.info(f"数据集分析完成: {stats['total_files']} 个文件, {stats['total_nodes']} 个节点, {stats['total_edges']} 条边")
        return stats
        
    except Exception as e:
        logger.error(f"分析CFG数据时出错: {e}")
        return {}


def main():
    """
    Main function for CFG feature extraction.
    """
    # Configuration paths (Ubuntu format)
    input_dir = "/mnt/data1_l20_raid5disk/lbq_dataset/dataset/benign_ida_analysis_cfg"
    output_dir = "/mnt/data1_l20_raid5disk/lbq_dataset/dataset/benign_cfg"
    fasttext_model_path = "/mnt/data1_l20_raid5disk/lbq_dataset/output/cfg_fasttext/finetuned_fasttext_model.bin"
    
    logger.info("开始CFG特征提取...")
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"FastText模型: {fasttext_model_path}")
    
    # 运行批量处理
    stats = batch_process_cfg(input_dir, output_dir, fasttext_model_path)
    
    logger.info(f"处理统计: {stats}")
    
    logger.info("CFG特征提取完成!")


if __name__ == "__main__":
    main()