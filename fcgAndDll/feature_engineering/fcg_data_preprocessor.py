#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FCG特征工程脚本
用于从FCG JSON文件构建GCN输入数据，包括特征矩阵X和关系矩阵edge_index
"""

import json
import numpy as np
import torch
import fasttext
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import torch
import fasttext
import numpy as np
from torch_geometric.data import Data
import concurrent.futures

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_function_name(func_name: str) -> Optional[str]:
    """
    清理函数名，移除@后缀，但保留 'sub_' 等内部函数名。
    
    Args:
        func_name: 原始函数名
        
    Returns:
        清理后的函数名
    """
    if not func_name:
        return None
        
    # 移除 @<数字> 后缀 (例如 _WinMain@16 -> _WinMain)
    cleaned_name = re.sub(r'@\d+$', '', func_name)
    
    # 移除函数名前的 _ 或 @ 符号
    # (例如 _memset -> memset, @GetProcAddress -> GetProcAddress)
    cleaned_name = cleaned_name.lstrip('_@')
    
    # 如果清理后为空，则返回None
    if not cleaned_name:
        return None

    # sub_401000 这类函数名会被保留，因为它们不匹配上述规则
    return cleaned_name


def build_node_mapping(fcg_data: Dict) -> Tuple[List[str], Dict[str, int]]:
    """
    构建节点列表和索引映射
    
    Args:
        fcg_data: FCG JSON数据
        
    Returns:
        (node_names, name_to_index): 节点名称列表和名称到索引的映射
    """
    # 从call_graph中获取所有函数名
    all_functions = set(fcg_data["fcg_data"]["call_graph"].keys())
    
    # 清理函数名并过滤无效的
    valid_functions = []
    for func_name in all_functions:
        cleaned_name = clean_function_name(func_name)
        if cleaned_name:
            valid_functions.append(cleaned_name)
    
    # 去重并排序，确保结果的一致性
    node_names = sorted(list(set(valid_functions)))
    name_to_index = {name: i for i, name in enumerate(node_names)}
    
    logger.info(f"构建了 {len(node_names)} 个有效节点")
    return node_names, name_to_index


def build_feature_matrix(node_names: List[str], fasttext_model) -> np.ndarray:
    """
    构建特征矩阵X，使用FastText模型生成300维向量
    
    Args:
        node_names: 节点名称列表
        fasttext_model: 预训练的FastText模型
        
    Returns:
        特征矩阵X，形状为(num_nodes, 300)
    """
    num_nodes = len(node_names)
    X = np.zeros((num_nodes, 300))
    
    for i, func_name in enumerate(node_names):
        # 使用FastText模型获取函数名的向量表示
        vector = fasttext_model.get_word_vector(func_name)
        X[i] = vector
    
    logger.info(f"构建了特征矩阵，形状: {X.shape}")
    return X


def build_edge_index(fcg_data: Dict, name_to_index: Dict[str, int]) -> torch.Tensor:
    """
    构建关系矩阵edge_index，包含去重逻辑
    
    Args:
        fcg_data: FCG JSON数据
        name_to_index: 名称到索引的映射
        
    Returns:
        关系矩阵edge_index，形状为(2, num_edges)
    """
    # 使用set自动去重
    edge_set = set()
    
    # 遍历所有函数调用
    function_calls = fcg_data["fcg_data"].get("function_calls", [])
    
    for call in function_calls:
        caller_name = call["caller"]
        callee_name = call["callee"]
        
        # 清理函数名
        cleaned_caller = clean_function_name(caller_name)
        cleaned_callee = clean_function_name(callee_name)
        
        # 查找索引
        caller_index = name_to_index.get(cleaned_caller)
        callee_index = name_to_index.get(cleaned_callee)
        
        # 检查有效性并添加到set
        if caller_index is not None and callee_index is not None:
            edge_set.add((caller_index, callee_index))
    
    # 转换为列表
    if edge_set:
        source_nodes = []
        target_nodes = []
        for src, tgt in edge_set:
            source_nodes.append(src)
            target_nodes.append(tgt)
        
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    else:
        # 如果没有边，创建空的edge_index
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    logger.info(f"构建了关系矩阵，边数: {edge_index.shape[1]}")
    return edge_index


def process_fcg_json(json_file_path: str, fasttext_model) -> Optional[Data]:
    """
    处理单个FCG JSON文件，生成图数据对象
    
    Args:
        json_file_path: JSON文件路径
        fasttext_model: FastText模型
        
    Returns:
        PyTorch Geometric Data对象，包含x和edge_index
    """
    try:
        # 加载JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            fcg_data = json.load(f)
        
        # 检查数据结构
        if "fcg_data" not in fcg_data or "call_graph" not in fcg_data["fcg_data"]:
            logger.warning(f"文件 {json_file_path} 缺少必要的数据结构")
            return None
        
        # 步骤1: 构建节点列表和索引映射
        node_names, name_to_index = build_node_mapping(fcg_data)
        
        if not node_names:
            logger.warning(f"文件 {json_file_path} 没有有效的函数节点")
            return None
        
        # 步骤2: 构建特征矩阵
        X = build_feature_matrix(node_names, fasttext_model)
        
        # 步骤3: 构建关系矩阵
        edge_index = build_edge_index(fcg_data, name_to_index)
        
        # 创建PyTorch Geometric Data对象
        data = Data(x=torch.tensor(X, dtype=torch.float32), edge_index=edge_index)
        
        logger.info(f"成功处理文件 {json_file_path}")
        return data
        
    except Exception as e:
        logger.error(f"处理文件 {json_file_path} 时出错: {str(e)}")
        return None


def process_and_save_one_file(json_file_path: Path, fasttext_model_path: str, output_dir: str) -> bool:
    """
    加载、处理单个FCG JSON文件，并保存为.pt文件
    
    Args:
        json_file_path (Path): 输入的JSON文件路径
        fasttext_model_path (str): fasttext模型的文件路径
        output_dir (str): .pt文件的输出目录
        
    Returns:
        bool: 处理是否成功
    """
    
    # 关键：每个进程必须独立加载模型
    # 因为模型对象不能在进程间安全传递
    try:
        fasttext_model = fasttext.load_model(fasttext_model_path)
    except Exception as e:
        logger.error(f"进程 {os.getpid()} 无法加载模型: {e}")
        return False

    logger.info(f"[PID {os.getpid()}] 开始处理文件: {json_file_path.name}")
    
    try:
        # 调用已有的处理函数
        data = process_fcg_json(str(json_file_path), fasttext_model)
        
        if data is not None:
            # 保存处理后的数据
            output_file = Path(output_dir) / f"{json_file_path.stem}_graph.pt"
            torch.save(data, output_file)
            logger.info(f"[PID {os.getpid()}] 成功保存: {output_file}")
            return True
        else:
            logger.warning(f"[PID {os.getpid()}] 处理失败 (data is None): {json_file_path.name}")
            return False
            
    except Exception as e:
        logger.error(f"[PID {os.getpid()}] 处理 {json_file_path.name} 时出错: {e}", exc_info=True)
        return False


def batch_process_fcg(input_dir: str, output_dir: str, fasttext_model_path: str, num_workers: int = None):
    """
    使用多进程批量处理所有FCG JSON文件
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
        fasttext_model_path (str): FastText模型路径
        num_workers (int, optional): 工作进程数量，默认为None（使用所有CPU核心）
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 收集所有任务
    input_path = Path(input_dir)
    json_files = list(input_path.glob("*_fcg.json"))
    
    if not json_files:
        logger.warning(f"在 {input_dir} 中未找到 *_fcg.json 文件")
        return

    logger.info(f"找到 {len(json_files)} 个FCG JSON文件，开始多进程处理...")
    
    # 2. 设置工作进程数
    if num_workers is None:
        num_workers = os.cpu_count()
    else:
        # 确保进程数不超过CPU核心数和文件数
        max_workers = min(os.cpu_count(), len(json_files))
        num_workers = min(num_workers, max_workers)
    
    logger.info(f"启动 {num_workers} 个工作进程（系统CPU核心数: {os.cpu_count()}）...")
    
    processed_count = 0
    failed_count = 0
    
    # 3. 使用 ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        # 提交所有任务到进程池
        futures = {
            executor.submit(process_and_save_one_file, json_file, fasttext_model_path, output_dir): json_file 
            for json_file in json_files
        }
        
        # 4. 收集结果
        for future in concurrent.futures.as_completed(futures):
            json_file = futures[future]
            try:
                success = future.result()
                if success:
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"文件 {json_file.name} 在执行中抛出异常: {e}")
                failed_count += 1
    
    logger.info(f"批量处理完成: 成功 {processed_count} 个，失败 {failed_count} 个")


def main():
    """主函数"""
    # 配置路径
    input_dir = "/mnt/data1_l20_raid5disk/lbq_dataset/dataset/benign_ida_fcg"
    fasttext_model_path = "/mnt/data1_l20_raid5disk/lbq_dataset/output/fcg_fasttext/finetuned_fasttext_model.bin"
    output_dir = "/mnt/data1_l20_raid5disk/lbq_dataset/dataset/benign_fcg"
    
    # 检查输入目录和模型文件是否存在
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    if not os.path.exists(fasttext_model_path):
        logger.error(f"FastText模型文件不存在: {fasttext_model_path}")
        return
    
    # 设置固定的工作进程数量
    num_workers = 16  # 写死使用16个进程
    print(f"系统CPU核心数: {os.cpu_count()}")
    logger.info(f"使用固定的 {num_workers} 个工作进程进行处理")
    
    # 批量处理
    batch_process_fcg(input_dir, output_dir, fasttext_model_path, num_workers)


if __name__ == "__main__":
    main()