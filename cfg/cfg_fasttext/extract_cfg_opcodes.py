#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFG操作码提取脚本
从benign_ida_analysis_cfg文件夹中的JSON文件提取基本块的操作码序列，
只保留semantic_map中定义的操作码，输出到corpus文件
"""

import json
import os
from pathlib import Path


def get_valid_opcodes():
    """获取semantic_map中定义的所有有效操作码"""
    semantic_map = {
        # Type 1: 基础控制类 (9个操作码)
        'type1_basic_control': {
            'leave', 'or', 'lock', 'and', 'call', 'test', 'add', 'not', 'ret'
        },
        
        # Type 2: 复合操作类 (53个操作码，包含4个子类)
        # 数学运算 (Math Operation) - 12个
        'type2_math_operations': {
            'mul', 'fmul', 'imul', 'adc', 'fadd', 'sub', 'sbb', 'inc', 'dec', 'cmp', 'idiv', 'neg'
        },
        
        # 逻辑跳转 (Logical Jump) - 23个
        'type2_logical_jump': {
            'jbe', 'jz', 'jge', 'jns', 'jb', 'jnb', 'jle', 'jnz', 'jmp', 'jg', 'js', 'ja', 'jl', 'nop', 'rep', 'lea', 'retn', 'retf', 'jo', 'jno', 'jp', 'jnp', 'jecxz'
        },
        
        # 数据操作 (Data Operation) - 14个
        'type2_data_operations': {
            'movdqa', 'movss', 'movd', 'mov', 'movsd', 'movsx', 'movdqu', 'movzx', 'movq', 'xchg', 'sar', 'sal', 'shr', 'shl'
        },
        
        # 栈操作 (Stack Operation) - 4个
        'type2_stack_operations': {
            'popf', 'pop', 'pushf', 'push'
        },
        
        # Type 3: 特殊指令类 (17个操作码)
        'type3_special_instructions': {
            'xor', 'fstp', 'fild', 'fld', 'fxch', 'cdq', 'setnz', 'setnle', 'setz', 'std', 'cldflag', 'fnclex', 'stosd', 'fnstcw', 'setl', 'int', 'wait'
        }
    }
    
    # 合并所有操作码到一个集合中
    valid_opcodes = set()
    for category_opcodes in semantic_map.values():
        valid_opcodes.update(category_opcodes)
    
    return valid_opcodes


def filter_opcodes(opcode_sequence, valid_opcodes):
    """过滤操作码序列，只保留有效的操作码"""
    if not opcode_sequence:
        return []
    
    filtered_opcodes = []
    for opcode in opcode_sequence:
        # filtered_opcodes.append(opcode.lower())
        if opcode and opcode.lower() in valid_opcodes:
            filtered_opcodes.append(opcode.lower())
    
    return filtered_opcodes


def extract_basic_blocks_from_json(json_file_path):
    """从JSON文件中提取所有基本块的操作码序列"""
    basic_blocks_opcodes = []  # 每个基本块为一个操作码列表
    valid_opcodes = get_valid_opcodes()
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cfg_data = data.get('cfg_data', {})
        global_basic_blocks = cfg_data.get('global_basic_blocks', [])
        
        if not global_basic_blocks:
            return []
        
        # 遍历所有基本块
        for basic_block in global_basic_blocks:
            opcode_sequence = basic_block.get('opcode_sequence', [])
            
            # 过滤操作码，只保留有效的
            filtered_opcodes = filter_opcodes(opcode_sequence, valid_opcodes)
            
            # 只有当基本块包含有效操作码时才添加
            if filtered_opcodes:
                basic_blocks_opcodes.append(filtered_opcodes)
        
    except Exception as e:
        print(f"Error processing {json_file_path}: {e}")
    
    return basic_blocks_opcodes


def main():
    # 设置路径
    input_dir = Path("/mnt/data1_l20_raid5disk/lbq_dataset/dataset/fasttext_data_controlFG")
    output_dir = Path("/mnt/data1_l20_raid5disk/lbq_dataset/output")
    output_file = output_dir / "cfg_corpus.txt"
    

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_files = 0
    total_basic_blocks = 0
    
    # 写入输出文件 - 每行代表一个基本块，操作码用空格分开
    with open(output_file, 'w', encoding='utf-8') as f:
        # 遍历所有JSON文件
        for json_file in input_dir.glob("*.json"):
            basic_blocks_opcodes = extract_basic_blocks_from_json(json_file)
            
            # 每个基本块写入一行
            for opcodes in basic_blocks_opcodes:
                if opcodes:  # 确保基本块不为空
                    opcode_line = " ".join(opcodes)
                    f.write(f"{opcode_line}\n")
                    total_basic_blocks += 1
            
            processed_files += 1
            
            if processed_files % 10 == 0:
                print(f"Processed {processed_files} files, extracted {total_basic_blocks} basic blocks...")
    
    print(f"Processed {processed_files} JSON files")
    print(f"Extracted {total_basic_blocks} basic blocks")
    print(f"Output saved to: {output_file}")
    print("Format: Each line represents one basic block, opcodes separated by spaces")
    
    # 统计有效操作码数量
    valid_opcodes = get_valid_opcodes()
    print(f"Total valid opcodes defined: {len(valid_opcodes)}")
    print(f"Type 1 (Basic Control): 9 opcodes")
    print(f"Type 2 (Composite Operations): 53 opcodes (Math:12 + Jump:23 + Data:14 + Stack:4)")
    print(f"Type 3 (Special Instructions): 17 opcodes")


if __name__ == "__main__":
    main()