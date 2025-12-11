#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FCG函数名提取脚本
从benign_ida_fcg文件夹中的JSON文件提取所有函数名，去重后输出到corpus文件
"""

import json
import os
import re
from pathlib import Path


def clean_function_name(func_name):
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


def extract_functions_from_json(json_file_path):
    """从JSON文件中提取函数名 - 按照call_graph中函数出现的顺序排列"""
    functions = []  # 使用列表保持顺序
    seen_functions = set()  # 用于去重
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fcg_data = data.get('fcg_data', {})
        call_graph = fcg_data.get('call_graph', {})
        
        if not call_graph:
            return []
        
        # 按照call_graph中函数出现的顺序提取函数
        for func_name in call_graph.keys():
            cleaned_name = clean_function_name(func_name)
            if cleaned_name and cleaned_name not in seen_functions:
                functions.append(cleaned_name)
                seen_functions.add(cleaned_name)
        
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
    
    return functions


def main():
    # 设置路径
    input_dir = Path("/mnt/data1_l20_raid5disk/lbq_dataset/dataset/fasttext_data_fcg")
    output_dir = Path("/mnt/data1_l20_raid5disk/lbq_dataset/output")
    output_file = output_dir / "fcg_corpus.txt"
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_files = 0
    
    # 写入输出文件 - 每行代表一个exe，函数用空格分开
    with open(output_file, 'w', encoding='utf-8') as f:
        # 遍历所有JSON文件
        for json_file in input_dir.glob("*.json"):
            functions = extract_functions_from_json(json_file)
            
            # 保持call_graph中的原始顺序，不再排序
            if functions:
                function_line = " ".join(functions)
                f.write(f"{function_line}\n")
            
            processed_files += 1
            
            if processed_files % 10 == 0:
                print(f"Processed {processed_files} files...")
    
    print(f"Processed {processed_files} JSON files")
    print(f"Output saved to: {output_file}")
    print("Format: Each line represents one executable, functions separated by spaces")


if __name__ == "__main__":
    main()