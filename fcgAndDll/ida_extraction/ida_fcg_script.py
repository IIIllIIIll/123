#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IDA Pro FCG提取脚本 - 在IDA环境中执行的函数调用图分析脚本
此脚本将在IDA Pro的Python环境中运行，专门用于提取函数调用图(FCG)
"""

import idaapi
import idautils
import idc
import json
import os
import time
import sys
import hashlib
import ida_nalt

def wait_for_analysis():
    """等待IDA完成自动分析"""
    print("Waiting for IDA analysis to complete...")
    idaapi.auto_wait()
    print("IDA analysis completed")

def get_import_functions():
    """获取所有导入函数的地址和名称"""
    import_functions = {}
    
    try:
        # 获取导入模块数量
        nimps = ida_nalt.get_import_module_qty()
        
        for i in range(nimps):
            module_name = ida_nalt.get_import_module_name(i)
            if not module_name:
                module_name = "<unnamed>"
            
            def imp_cb(ea, name, ordinal):
                if name:
                    # 存储导入函数的地址和完整名称
                    import_functions[ea] = f"{module_name}!{name}"
                else:
                    import_functions[ea] = f"{module_name}!ordinal#{ordinal}"
                return True  # 继续枚举
            
            ida_nalt.enum_import_names(i, imp_cb)
        
        print(f"Found {len(import_functions)} import functions")
        return import_functions
        
    except Exception as e:
        print(f"Error getting import functions: {e}")
        return {}

def is_import_function(ea, import_functions):
    """检查给定地址是否为导入函数"""
    return ea in import_functions

def is_library_function(ea):
    """检查给定地址是否为库函数（使用IDA标志）"""
    try:
        func_flags = idc.get_func_flags(ea)
        return bool(func_flags & (idc.FUNC_LIB | idc.FUNC_THUNK))
    except:
        return False

def get_file_hash(file_path):
    """计算文件的SHA256哈希值"""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256()
            while chunk := f.read(8192):
                file_hash.update(chunk)
            return file_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating file hash: {e}")
        return None

def extract_basic_info():
    """提取PE文件基本信息"""
    try:
        input_file = idaapi.get_input_file_path()
        file_hash = get_file_hash(input_file)
        
        info = {
            "filename": os.path.basename(input_file),
            "filepath": input_file,
            "file_hash": file_hash,
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

def extract_function_calls():
    """提取函数调用关系"""
    function_calls = []
    call_graph = {}
    
    try:
        print("Extracting function call relationships...")
        
        # 获取导入函数信息
        import_functions = get_import_functions()
        
        # 获取所有函数
        functions = {}
        for func_ea in idautils.Functions():
            func_name = idc.get_func_name(func_ea)
            functions[func_ea] = func_name
            call_graph[func_name] = {
                "address": hex(func_ea),
                "calls": [],
                "called_by": []
            }
        
        # 分析每个函数的调用关系
        for func_ea in idautils.Functions():
            func_name = idc.get_func_name(func_ea)
            func = idaapi.get_func(func_ea)
            
            if not func:
                continue
            
            # 遍历函数中的所有指令
            ea = func.start_ea
            while ea < func.end_ea:
                insn = idaapi.insn_t()
                if idaapi.decode_insn(insn, ea):
                    # 检查是否是调用指令
                    if idaapi.is_call_insn(insn):
                        # 获取调用目标
                        target_ea = insn.ops[0].addr if insn.ops[0].type == idaapi.o_near else None
                        
                        if target_ea:
                            # 确定目标函数名称
                            if target_ea in functions:
                                target_func_name = functions[target_ea]
                            else:
                                # 可能是外部函数或导入函数
                                target_func_name = idc.get_name(target_ea)
                                if not target_func_name:
                                    target_func_name = f"sub_{target_ea:X}"
                            
                            # 检查是否为导入函数或库函数
                            is_import = (is_import_function(target_ea, import_functions) or 
                                       is_library_function(target_ea))
                            
                            # 记录调用关系
                            call_info = {
                                "caller": func_name,
                                "caller_address": hex(func_ea),
                                "callee": target_func_name,
                                "callee_address": hex(target_ea),
                                "call_site": hex(ea),
                                "is_import": is_import
                            }
                            
                            function_calls.append(call_info)
                            
                            # 更新调用图（仅对内部函数）
                            if target_ea in functions:
                                if target_func_name not in call_graph[func_name]["calls"]:
                                    call_graph[func_name]["calls"].append(target_func_name)
                                
                                if func_name not in call_graph[target_func_name]["called_by"]:
                                    call_graph[target_func_name]["called_by"].append(func_name)
                    
                    ea += insn.size
                else:
                    ea += 1
        
        print(f"Function calls extracted: {len(function_calls)} call relationships found")
        return function_calls, call_graph
        
    except Exception as e:
        print(f"Error extracting function calls: {e}")
        return [], {}

def extract_fcg_features():
    """提取FCG特征"""
    try:
        print("Extracting FCG features...")
        
        function_calls, call_graph = extract_function_calls()
        
        # 计算FCG统计特征
        total_functions = len(call_graph)
        total_calls = len(function_calls)
        
        # 计算入度和出度统计
        in_degrees = []
        out_degrees = []
        
        for func_name, func_info in call_graph.items():
            in_degree = len(func_info["called_by"])
            out_degree = len(func_info["calls"])
            in_degrees.append(in_degree)
            out_degrees.append(out_degree)
        
        # 计算统计信息
        avg_in_degree = sum(in_degrees) / max(1, len(in_degrees))
        avg_out_degree = sum(out_degrees) / max(1, len(out_degrees))
        max_in_degree = max(in_degrees) if in_degrees else 0
        max_out_degree = max(out_degrees) if out_degrees else 0
        
        # 找出关键函数（高入度或高出度）
        key_functions = []
        for func_name, func_info in call_graph.items():
            in_degree = len(func_info["called_by"])
            out_degree = len(func_info["calls"])
            
            if in_degree > avg_in_degree * 2 or out_degree > avg_out_degree * 2:
                key_functions.append({
                    "name": func_name,
                    "address": func_info["address"],
                    "in_degree": in_degree,
                    "out_degree": out_degree
                })
        
        fcg_features = {
            "function_calls": function_calls,
            "call_graph": call_graph,
            "statistics": {
                "total_functions": total_functions,
                "total_calls": total_calls,
                "avg_in_degree": avg_in_degree,
                "avg_out_degree": avg_out_degree,
                "max_in_degree": max_in_degree,
                "max_out_degree": max_out_degree,
                "density": total_calls / max(1, total_functions * (total_functions - 1)) if total_functions > 1 else 0
            },
            "key_functions": key_functions
        }
        
        print(f"FCG features extracted: {total_functions} functions, {total_calls} calls")
        return fcg_features
        
    except Exception as e:
        print(f"Error extracting FCG features: {e}")
        return {"error": str(e)}

def extract_api_calls():
    """提取API调用信息"""
    api_calls = []
    
    try:
        print("Extracting API calls...")
        
        # 获取导入函数信息
        import_functions = get_import_functions()
        
        for func_ea in idautils.Functions():
            func_name = idc.get_func_name(func_ea)
            func = idaapi.get_func(func_ea)
            
            if not func:
                continue
            
            # 遍历函数中的所有指令
            ea = func.start_ea
            while ea < func.end_ea:
                insn = idaapi.insn_t()
                if idaapi.decode_insn(insn, ea):
                    # 检查是否是调用指令
                    if idaapi.is_call_insn(insn):
                        # 获取调用目标
                        target_ea = insn.ops[0].addr if insn.ops[0].type == idaapi.o_near else None
                        
                        if target_ea:
                            # 获取调用目标名称
                            target_name = idc.get_name(target_ea)
                            if not target_name:
                                target_name = idc.print_operand(ea, 0)
                            
                            # 检查是否为导入函数或库函数
                            is_import = (is_import_function(target_ea, import_functions) or 
                                       is_library_function(target_ea))
                            
                            # 如果是导入函数或包含API关键词，则认为是API调用
                            if (is_import or 
                                any(keyword in target_name.lower() for keyword in ['api', 'dll', 'kernel32', 'user32', 'ntdll', 'advapi32'])):
                                
                                api_call = {
                                    "caller_function": func_name,
                                    "caller_address": hex(func_ea),
                                    "api_name": target_name,
                                    "call_site": hex(ea),
                                    "is_import": is_import
                                }
                                api_calls.append(api_call)
                    
                    ea += insn.size
                else:
                    ea += 1
        
        print(f"API calls extracted: {len(api_calls)} API calls found")
        return api_calls
        
    except Exception as e:
        print(f"Error extracting API calls: {e}")
        return []

def main():
    """主函数"""
    print("Starting IDA Pro FCG extraction script...")
    
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
        
        # 提取FCG特征
        print("Extracting FCG features...")
        fcg_features = extract_fcg_features()
        
        # 提取API调用
        print("Extracting API calls...")
        api_calls = extract_api_calls()
        
        # 组合结果
        results = {
            "basic_info": basic_info,
            "fcg_data": fcg_features,
            "api_calls": api_calls,
            "analysis_timestamp": time.time(),
            "analysis_type": "FCG"
        }
        
        # 使用文件名作为输出文件名，不需要重新计算哈希
        input_file = idaapi.get_input_file_path()
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        output_filename = f"{file_name}_fcg.json"
        
        # 保存到JSON文件
        output_file = os.path.join(output_dir, output_filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"FCG analysis completed successfully!")
        print(f"Results saved to: {output_file}")
        
        # 输出统计信息
        if "statistics" in fcg_features:
            stats = fcg_features["statistics"]
            print(f"Statistics:")
            print(f"  - Total functions: {stats['total_functions']}")
            print(f"  - Total calls: {stats['total_calls']}")
            print(f"  - Average in-degree: {stats['avg_in_degree']:.2f}")
            print(f"  - Average out-degree: {stats['avg_out_degree']:.2f}")
            print(f"  - Graph density: {stats['density']:.4f}")
        
    except Exception as e:
        print(f"FCG analysis failed: {e}")
        # 保存错误信息
        try:
            output_dir = os.environ.get('IDA_OUTPUT_DIR', os.getcwd())
            error_file = os.path.join(output_dir, f"fcg_error_{int(time.time())}.json")
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump({"error": str(e), "timestamp": time.time(), "analysis_type": "FCG"}, f)
            print(f"Error info saved to: {error_file}")
        except:
            pass
    
    finally:
        # 退出IDA
        print("Exiting IDA Pro...")
        idaapi.qexit(0)

if __name__ == "__main__":
    main()