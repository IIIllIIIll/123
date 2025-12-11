#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 .npy 文件内容的工具
用于检查和分析 benign_RGB 目录中的 .npy 文件
"""

import numpy as np
import os
import sys
from pathlib import Path

def read_npy_file(file_path):
    """
    读取 .npy 文件并返回其内容和基本信息
    
    Args:
        file_path (str): .npy 文件的路径
        
    Returns:
        dict: 包含文件信息的字典
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return {
                'file_path': file_path,
                'exists': False,
                'error': '文件不存在',
                'data': None,
                'shape': None,
                'dtype': None,
                'size': None
            }
        
        # 读取 .npy 文件
        data = np.load(file_path)
        
        return {
            'file_path': file_path,
            'exists': True,
            'error': None,
            'data': data,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'size': data.size,
            'min_value': np.min(data) if data.size > 0 else None,
            'max_value': np.max(data) if data.size > 0 else None,
            'mean_value': np.mean(data) if data.size > 0 else None,
            'std_value': np.std(data) if data.size > 0 else None
        }
        
    except Exception as e:
        return {
            'file_path': file_path,
            'exists': True,
            'error': f'读取文件时出错: {str(e)}',
            'data': None,
            'shape': None,
            'dtype': None,
            'size': None
        }

def read_multiple_npy_files(file_list, base_dir=None):
    """
    读取多个 .npy 文件
    
    Args:
        file_list (list): 文件名列表
        base_dir (str): 基础目录路径
        
    Returns:
        list: 包含所有文件信息的列表
    """
    results = []
    
    for filename in file_list:
        if base_dir:
            file_path = os.path.join(base_dir, filename)
        else:
            file_path = filename
            
        result = read_npy_file(file_path)
        results.append(result)
        
    return results

def print_file_info(file_info):
    """
    打印文件信息
    
    Args:
        file_info (dict): 文件信息字典
    """
    print(f"\n文件: {os.path.basename(file_info['file_path'])}")
    print(f"路径: {file_info['file_path']}")
    print(f"存在: {file_info['exists']}")
    
    if file_info['error']:
        print(f"错误: {file_info['error']}")
        return
    
    if file_info['exists'] and file_info['data'] is not None:
        print(f"形状: {file_info['shape']}")
        print(f"数据类型: {file_info['dtype']}")
        print(f"元素数量: {file_info['size']}")
        
        if file_info['min_value'] is not None:
            print(f"最小值: {file_info['min_value']}")
            print(f"最大值: {file_info['max_value']}")
            print(f"平均值: {file_info['mean_value']:.6f}")
            print(f"标准差: {file_info['std_value']:.6f}")
        
        # 显示数据的前几个元素（如果是一维数组）
        if len(file_info['shape']) == 1 and file_info['size'] <= 20:
            print(f"数据内容: {file_info['data']}")
        elif len(file_info['shape']) == 1 and file_info['size'] > 20:
            print(f"数据前10个元素: {file_info['data'][:10]}")
            print(f"数据后10个元素: {file_info['data'][-10:]}")
        elif len(file_info['shape']) == 2:
            print(f"数据形状为2D，前5行:")
            print(file_info['data'][:5])

def main():
    """
    主函数 - 读取用户提到的特定文件
    """
    # 用户提到的文件列表
    target_files = [
        "b88f2fecb75a2167659a411eee93122514d0c90d33b52bcbbf93799ec31e1d35.exe.npy",
        "25aef89003c3a48297cc4e787d63901d63cc837052ea877b48a4842011fe989d.exe.npy",
        "c86efc621cd81eb1f914d94f5e18d2405186f622878eea63ebf23938e9963b7f.exe.npy",
        "4143ff84431b630fb19c0e60611c03bb453e4a9f88fa551f245e9bf7d8741cca.exe.npy",
        "28a720b196843d3fa57dee4864ad67e6da1f43c9ec07326b3e094fb15e4f87f8.exe.npy",
        "7e8e055425aba46a519fadeea02d60065081342ad3a3015fddb952b170b999da.exe.npy"
    ]
    
    # 基础目录
    base_dir = r"d:\Test\AAAAAAAAAAAAAAA\demooooooo\dataset\benign_RGB"
    
    print("开始读取 .npy 文件...")
    print(f"基础目录: {base_dir}")
    print("=" * 80)
    
    # 读取所有文件
    results = read_multiple_npy_files(target_files, base_dir)
    
    # 打印结果
    for result in results:
        print_file_info(result)
        print("-" * 60)
    
    # 统计信息
    existing_files = [r for r in results if r['exists'] and not r['error']]
    print(f"\n总结:")
    print(f"总文件数: {len(results)}")
    print(f"存在的文件数: {len([r for r in results if r['exists']])}")
    print(f"成功读取的文件数: {len(existing_files)}")
    
    if existing_files:
        shapes = [r['shape'] for r in existing_files]
        dtypes = [r['dtype'] for r in existing_files]
        print(f"文件形状: {set(shapes)}")
        print(f"数据类型: {set(dtypes)}")

if __name__ == "__main__":
    main()