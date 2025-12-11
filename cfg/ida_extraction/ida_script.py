#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IDA Pro脚本 - 在IDA环境中执行的分析脚本
此脚本将在IDA Pro的Python环境中运行
"""

import idaapi
import idautils
import idc
import json
import os
import time
import sys

def wait_for_analysis():
    """等待IDA完成自动分析"""
    print("Waiting for IDA analysis to complete...")
    idaapi.auto_wait()
    print("IDA analysis completed")

def extract_basic_info():
    """提取PE文件基本信息"""
    try:
        info = {
            "filename": idaapi.get_input_file_path(),
            "file_type": "PE",
            "architecture": "x64" if idc.get_inf_attr(idc.INF_LFLAGS) & idc.LFLG_64BIT else "x86",
            "entry_point": hex(idc.get_inf_attr(idc.INF_START_EA)),
            "image_base": hex(idc.get_inf_attr(idc.INF_MIN_EA)),
            "max_address": hex(idc.get_inf_attr(idc.INF_MAX_EA)),
            "functions_count": len(list(idautils.Functions()))
        }
        print(f"Basic info extracted: {info['functions_count']} functions found")
        return info
    except Exception as e:
        print(f"Error extracting basic info: {e}")
        return {"error": str(e)}

def extract_opcode_features():
    """提取操作码特征"""
    important_opcodes = [
        'mov', 'push', 'pop', 'call', 'ret', 'jmp', 'je', 'jne', 'jz', 'jnz',
        'add', 'sub', 'mul', 'div', 'inc', 'dec', 'and', 'or', 'xor', 'not',
        'shl', 'shr', 'cmp', 'test', 'lea', 'nop', 'int', 'syscall', 'sysenter',
        'leave', 'enter', 'pusha', 'popa', 'pushf', 'popf', 'cli', 'sti',
        'rep', 'repe', 'repne', 'loop', 'loope', 'loopne', 'jl', 'jle', 'jg', 'jge',
        'ja', 'jae', 'jb', 'jbe', 'jo', 'jno', 'js', 'jns', 'jp', 'jnp',
        'lahf', 'sahf', 'cbw', 'cwd', 'cdq', 'movsx', 'movzx', 'bswap',
        'xchg', 'cmpxchg', 'lock', 'hlt', 'wait', 'fwait', 'rdtsc'
    ]
    
    opcode_counts = {op: 0 for op in important_opcodes}
    opcode_counts['other'] = 0
    total_instructions = 0
    
    try:
        print("Extracting opcode features...")
        for func_ea in idautils.Functions():
            func = idaapi.get_func(func_ea)
            if not func:
                continue
                
            ea = func.start_ea
            while ea < func.end_ea:
                insn = idaapi.insn_t()
                if idaapi.decode_insn(insn, ea):
                    mnem = idc.print_insn_mnem(ea).lower()
                    total_instructions += 1
                    
                    if mnem in opcode_counts:
                        opcode_counts[mnem] += 1
                    else:
                        opcode_counts['other'] += 1
                    
                    ea += insn.size
                else:
                    ea += 1
        
        # 计算频率
        if total_instructions > 0:
            opcode_frequencies = {op: count/total_instructions for op, count in opcode_counts.items()}
        else:
            opcode_frequencies = opcode_counts
            
        print(f"Opcode features extracted: {total_instructions} instructions analyzed")
        return {
            "opcode_counts": opcode_counts,
            "opcode_frequencies": opcode_frequencies,
            "total_instructions": total_instructions
        }
    except Exception as e:
        print(f"Error extracting opcode features: {e}")
        return {"error": str(e)}

def extract_cfg_data():
    """提取CFG数据"""
    cfg_data = {
        "functions": [],
        "global_basic_blocks": [],
        "global_edges": [],
        "statistics": {}
    }
    
    try:
        print("Extracting CFG data...")
        global_bb_id = 0
        bb_id_mapping = {}
        
        for func_ea in idautils.Functions():
            func = idaapi.get_func(func_ea)
            if not func:
                continue
                
            func_name = idc.get_func_name(func_ea)
            flowchart = idaapi.FlowChart(func)
            
            func_data = {
                "name": func_name,
                "address": hex(func_ea),
                "start": hex(func.start_ea),
                "end": hex(func.end_ea),
                "basic_blocks": [],
                "local_edges": []
            }
            
            # 处理基本块
            for bb in flowchart:
                bb_start = bb.start_ea
                bb_end = bb.end_ea
                
                bb_id_mapping[bb_start] = global_bb_id
                
                # 计算基本块特征
                instruction_count = 0
                opcode_vector = {}
                opcode_sequence = []  # 新增：记录操作码的具体顺序
                
                ea = bb_start
                while ea < bb_end:
                    insn = idaapi.insn_t()
                    if idaapi.decode_insn(insn, ea):
                        instruction_count += 1
                        mnem = idc.print_insn_mnem(ea).lower()
                        opcode_vector[mnem] = opcode_vector.get(mnem, 0) + 1
                        opcode_sequence.append(mnem)  # 新增：按顺序记录每个操作码
                        ea += insn.size
                    else:
                        ea += 1
                
                bb_data = {
                    "global_id": global_bb_id,
                    "function": func_name,
                    "start": hex(bb_start),
                    "end": hex(bb_end),
                    "size": bb_end - bb_start,
                    "instruction_count": instruction_count,
                    "opcode_vector": opcode_vector,
                    "opcode_sequence": opcode_sequence  # 新增：操作码的具体顺序
                }
                
                cfg_data["global_basic_blocks"].append(bb_data)
                func_data["basic_blocks"].append(global_bb_id)
                global_bb_id += 1
            
            # 处理边
            for bb in flowchart:
                source_id = bb_id_mapping[bb.start_ea]
                for succ in bb.succs():
                    target_id = bb_id_mapping[succ.start_ea]
                    
                    edge_data = {
                        "source": source_id,
                        "target": target_id,
                        "function": func_name,
                        "type": "control_flow"
                    }
                    
                    cfg_data["global_edges"].append(edge_data)
                    func_data["local_edges"].append(edge_data)
            
            cfg_data["functions"].append(func_data)
        
        # 统计信息
        cfg_data["statistics"] = {
            "total_functions": len(cfg_data["functions"]),
            "total_basic_blocks": len(cfg_data["global_basic_blocks"]),
            "total_edges": len(cfg_data["global_edges"]),
            "avg_bb_per_function": len(cfg_data["global_basic_blocks"]) / max(1, len(cfg_data["functions"]))
        }
        
        print(f"CFG data extracted: {cfg_data['statistics']['total_functions']} functions, {cfg_data['statistics']['total_basic_blocks']} basic blocks")
        return cfg_data
        
    except Exception as e:
        print(f"Error extracting CFG data: {e}")
        return {"error": str(e)}

def main():
    """主函数"""
    print("Starting IDA Pro analysis script...")
    
    try:
        # 等待分析完成
        wait_for_analysis()
        
        # 获取输出目录（从环境变量或命令行参数）
        output_dir = os.environ.get('IDA_OUTPUT_DIR', os.getcwd())
        
        print(f"Output directory: {output_dir}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 提取基本信息
        print("Extracting basic info...")
        basic_info = extract_basic_info()
        
        # 提取操作码特征
        print("Extracting opcode features...")
        opcode_features = extract_opcode_features()
        
        # 提取CFG数据
        print("Extracting CFG data...")
        cfg_data = extract_cfg_data()
        
        # 组合结果
        results = {
            "basic_info": basic_info,
            "opcode_features": opcode_features,
            "cfg_data": cfg_data,
            "analysis_timestamp": time.time()
        }
        
        # 获取文件名
        input_file = idaapi.get_input_file_path()
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # 保存到JSON文件
        output_file = os.path.join(output_dir, f"{file_name}_ida_analysis.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        # 保存错误信息
        try:
            output_dir = os.environ.get('IDA_OUTPUT_DIR', os.getcwd())
            error_file = os.path.join(output_dir, f"error_{int(time.time())}.json")
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({"error": str(e), "timestamp": time.time()}, f)
            print(f"Error info saved to: {error_file}")
        except:
            pass
    
    finally:
        # 退出IDA
        print("Exiting IDA Pro...")
        idaapi.qexit(0)

if __name__ == "__main__":
    main()