#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PE结构分析脚本
功能：
1. 检测指定文件夹内所有文件是否为PE结构
2. 统计每个PE文件的DLL导入个数
3. 生成JSON汇总报告，记录DLL个数为1和不是PE结构的文件
4. 记录文件名不是SHA256格式的文件
"""

import os
import json
import pefile
import signal
import sys
import re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class PEAnalyzer:
    def __init__(self, target_directory, output_file="pe_analysis_report.json", max_workers=8):
        """
        初始化PE分析器
        
        Args:
            target_directory (str): 目标文件夹路径
            output_file (str): 输出JSON文件名
            max_workers (int): 最大线程数
        """
        self.target_directory = Path(target_directory)
        self.output_file = self.target_directory / output_file
        self.max_workers = max_workers
        self.interrupted = False  # 中断标志
        
        # 统计数据
        self.total_files = 0
        self.valid_pe_files = 0
        self.invalid_pe_files = 0
        self.non_sha256_files = 0
        self.no_dll_pe_files = 0  # 无DLL导入的PE文件数量
        self.unanalyzable_files = 0  # 无法分析的文件数量
        self.processed_count = 0
        
        # 分析结果存储
        self.analysis_results = {
            "analysis_info": {
                "target_directory": str(self.target_directory),
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_files": 0,
                "valid_pe_files": 0,
                "invalid_pe_files": 0,
                "non_sha256_files": 0,
                "no_dll_pe_files": 0,
                "unanalyzable_files": 0
            },
            "invalid_pe_files": [],  # 不是PE结构的文件
            "non_sha256_files": [],  # 文件名不是SHA256格式的文件
            "no_dll_pe_files": [],   # 是PE文件但无DLL导入的文件
            "unanalyzable_files": []  # 无法分析的文件
        }
        
        # 线程锁，用于保护共享变量
        self.lock = threading.Lock()
        
        # 确保目标目录存在
        if not self.target_directory.exists():
            raise ValueError(f"目标目录不存在: {target_directory}")
        
        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理Ctrl+C信号"""
        print("\n接收到中断信号，正在停止处理...")
        self.interrupted = True
    
    def is_sha256_filename(self, filename):
        """
        检查文件名是否为SHA256格式（64位十六进制字符）
        
        Args:
            filename (str): 文件名（不包含扩展名）
            
        Returns:
            bool: 是否为SHA256格式
        """
        # 移除文件扩展名，只检查主文件名
        name_without_ext = Path(filename).stem
        
        # SHA256是64位十六进制字符
        sha256_pattern = r'^[a-fA-F0-9]{64}$'
        return bool(re.match(sha256_pattern, name_without_ext))
    
    def is_valid_pe_file(self, file_path):
        """
        检查文件是否为有效的PE文件
        
        Args:
            file_path (Path): 文件路径
            
        Returns:
            bool: 是否为有效PE文件
        """
        try:
            pe = pefile.PE(str(file_path))
            pe.close()
            return True
        except pefile.PEFormatError:
            return False
        except Exception:
            return False
    
    def get_dll_imports_info(self, file_path):
        """
        获取PE文件的DLL导入信息
        
        Args:
            file_path (Path): 文件路径
            
        Returns:
            tuple: (是否能读取DLL导入库, DLL数量, DLL列表)
        """
        try:
            pe = pefile.PE(str(file_path))
            
            # 检查是否有导入表
            if not hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                pe.close()
                return False, 0, []
            
            # 统计导入的DLL信息
            dll_list = []
            dll_count = 0
            
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('utf-8', errors='ignore')
                dll_count += 1
                
                # 统计该DLL中的函数数量
                function_count = 0
                functions = []
                for imp in entry.imports:
                    if imp.name:
                        function_count += 1
                        functions.append(imp.name.decode('utf-8', errors='ignore'))
                
                dll_info = {
                    "dll_name": dll_name,
                    "function_count": function_count,
                    "functions": functions[:10]  # 只保存前10个函数名，避免JSON过大
                }
                dll_list.append(dll_info)
            
            pe.close()
            return dll_count > 0, dll_count, dll_list
            
        except Exception as e:
            return False, 0, []
    
    def analyze_single_file(self, file_path):
        """
        分析单个文件
        
        Args:
            file_path (Path): 文件路径
            
        Returns:
            dict: 分析结果
        """
        # 检查是否被中断
        if self.interrupted:
            return None
            
        # 确保file_path是Path对象
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        filename = file_path.name
        
        # 首先尝试获取文件大小，如果失败则记录为无法分析
        try:
            file_size = file_path.stat().st_size
        except (OSError, IOError, PermissionError) as e:
            # 无法访问文件，记录为无法分析
            with self.lock:
                self.unanalyzable_files += 1
                self.analysis_results["unanalyzable_files"].append({
                    "filename": filename,
                    "size": 0,
                    "reason": f"无法访问文件: {str(e)}"
                })
            print(f"无法访问文件: {filename} - {e}")
            with self.lock:
                self.processed_count += 1
            return None
        
        result = {
            "filename": filename,
            "size": file_size,
            "is_pe": False,
            "dll_count": 0,
            "dll_imports": [],
            "analysis_status": "success",
            "is_sha256_filename": False
        }
        
        # 检查文件名是否为SHA256格式
        is_sha256 = self.is_sha256_filename(filename)
        result["is_sha256_filename"] = is_sha256
        
        # 如果不是SHA256格式，记录到non_sha256_files
        if not is_sha256:
            with self.lock:
                self.non_sha256_files += 1
                self.analysis_results["non_sha256_files"].append({
                    "filename": filename,
                    "size": file_size,
                    "reason": "文件名不符合SHA256格式"
                })
            print(f"非SHA256文件名: {filename}")
        
        try:
            # 检查是否为有效PE文件
            if not self.is_valid_pe_file(file_path):
                result["analysis_status"] = "not_pe_file"
                with self.lock:
                    self.invalid_pe_files += 1
                    self.analysis_results["invalid_pe_files"].append({
                    "filename": filename,
                    "size": file_size,
                    "reason": "不是有效的PE文件"
                })
                print(f"非PE文件: {filename}")
                with self.lock:
                    self.processed_count += 1
                return result
            
            result["is_pe"] = True
            
            # 获取DLL导入信息
            can_read, dll_count, dll_list = self.get_dll_imports_info(file_path)
            
            if not can_read:
                result["analysis_status"] = "no_dll_imports"
                result["dll_count"] = 0
                # 记录无DLL导入的PE文件
                with self.lock:
                    self.no_dll_pe_files += 1
                    self.analysis_results["no_dll_pe_files"].append({
                        "filename": filename,
                        "size": file_size,
                        "reason": "PE文件但无DLL导入"
                    })
                print(f"PE文件但无DLL导入: {filename}")
            else:
                result["dll_count"] = dll_count
                result["dll_imports"] = dll_list
                
                print(f"PE文件: {filename} (DLL导入数: {dll_count})")
            
            with self.lock:
                self.valid_pe_files += 1
            
        except Exception as e:
            # 分析过程中出现异常，记录为无法分析
            result["analysis_status"] = f"analysis_error: {str(e)}"
            with self.lock:
                self.unanalyzable_files += 1
                self.analysis_results["unanalyzable_files"].append({
                    "filename": filename,
                    "size": file_size,
                    "reason": f"分析异常: {str(e)}"
                })
            print(f"分析异常 {filename}: {e}")
        
        with self.lock:
            self.processed_count += 1
        
        return result
    
    def analyze_directory(self):
        """
        分析整个目录（多线程版本）
        """
        print(f"开始分析目录: {self.target_directory}")
        print(f"输出文件: {self.output_file}")
        print(f"使用线程数: {self.max_workers}")
        print("-" * 50)
        
        # 获取所有文件
        files = [f for f in self.target_directory.iterdir() if f.is_file()]
        self.total_files = len(files)
        
        if self.total_files == 0:
            print("目录中没有找到文件")
            return
        
        print(f"找到 {self.total_files} 个文件")
        
        # 使用多线程分析文件
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_file = {executor.submit(self.analyze_single_file, file_path): file_path 
                                 for file_path in files}
                
                # 处理完成的任务
                completed = 0
                for future in as_completed(future_to_file):
                    # 检查是否被中断
                    if self.interrupted:
                        print("\n正在取消剩余任务...")
                        # 取消所有未完成的任务
                        for f in future_to_file:
                            if not f.done():
                                f.cancel()
                        break
                        
                    completed += 1
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            print(f"[{completed}/{self.total_files}] 完成分析: {file_path.name}")
                    except Exception as exc:
                        print(f"[{completed}/{self.total_files}] 分析异常 {file_path.name}: {exc}")
        except KeyboardInterrupt:
            print("\n用户中断操作，正在停止处理...")
            self.interrupted = True
            return
        
        # 检查是否被中断，如果被中断则保存当前结果
        if self.interrupted:
            print("\n检测到中断信号，正在保存当前分析结果...")
            # 更新统计信息
            self.analysis_results["analysis_info"].update({
                "total_files": self.total_files,
                "valid_pe_files": self.valid_pe_files,
                "invalid_pe_files": self.invalid_pe_files,
                "non_sha256_files": self.non_sha256_files,
                "no_dll_pe_files": self.no_dll_pe_files,
                "unanalyzable_files": self.unanalyzable_files,
                "interrupted": True,
                "processed_files": self.processed_count
            })
            # 保存分析结果
            self.save_results()
            # 输出统计信息
            self.print_summary()
            return
        
        # 更新统计信息
        self.analysis_results["analysis_info"].update({
            "total_files": self.total_files,
            "valid_pe_files": self.valid_pe_files,
            "invalid_pe_files": self.invalid_pe_files,
            "non_sha256_files": self.non_sha256_files,
            "no_dll_pe_files": self.no_dll_pe_files,
            "unanalyzable_files": self.unanalyzable_files
        })
        
        # 保存分析结果
        self.save_results()
        
        # 输出统计信息
        self.print_summary()
    
    def save_results(self):
        """
        保存分析结果到JSON文件
        """
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, ensure_ascii=False, indent=2)
            print(f"\n分析结果已保存到: {self.output_file}")
        except Exception as e:
            print(f"保存分析结果失败: {e}")
    
    def print_summary(self):
        """
        打印分析结果摘要
        """
        print("\n" + "=" * 60)
        print("PE结构分析完成！统计信息：")
        print(f"总文件数: {self.total_files}")
        print(f"无效PE文件数: {self.invalid_pe_files}")
        print(f"非SHA256文件名数: {self.non_sha256_files}")
        print(f"无DLL导入PE文件数: {self.no_dll_pe_files}")
        print(f"无法分析文件数: {self.unanalyzable_files}")
        print(f"已处理文件数: {self.processed_count}")
        
        # 显示各类文件的占比
        if self.invalid_pe_files > 0:
            invalid_ratio = (self.invalid_pe_files / self.total_files) * 100
            print(f"无效PE文件占比: {invalid_ratio:.2f}%")
        
        if self.non_sha256_files > 0:
            non_sha256_ratio = (self.non_sha256_files / self.total_files) * 100
            print(f"非SHA256文件名占比: {non_sha256_ratio:.2f}%")
        
        if self.no_dll_pe_files > 0:
            no_dll_ratio = (self.no_dll_pe_files / self.total_files) * 100
            print(f"无DLL导入PE文件占比: {no_dll_ratio:.2f}%")
        
        if self.unanalyzable_files > 0:
            unanalyzable_ratio = (self.unanalyzable_files / self.total_files) * 100
            print(f"无法分析文件占比: {unanalyzable_ratio:.2f}%")
        
        # 检查是否被中断
        if hasattr(self, 'interrupted') and self.interrupted:
            print(f"\n注意: 分析被用户中断，仅处理了 {self.processed_count}/{self.total_files} 个文件")
        
        print(f"\n详细分析结果已保存到: {self.output_file}")
        print("=" * 60)


def main():
    """
    主函数：交互式运行或直接配置运行
    """
    # 交互式输入
    try:
        target_directory = input("请输入要分析的目录路径: ").strip()
        if not target_directory:
            # 如果没有输入，使用默认路径
            target_directory = r"D:\Test\AAAAAAAAAAAAAAA\demooooooo\dataset\malware"
            print(f"使用默认目录: {target_directory}")
        
        output_file = input("请输入输出JSON文件名 (默认pe_analysis_report.json): ").strip()
        if not output_file:
            output_file = "pe_analysis_report.json"
            print(f"使用默认输出文件: {output_file}")
        
        max_workers_input = input("请输入线程数 (默认8): ").strip()
        if max_workers_input and max_workers_input.isdigit():
            max_workers = int(max_workers_input)
        else:
            max_workers = 8
            print(f"使用默认线程数: {max_workers}")
        
        analyzer = PEAnalyzer(target_directory, output_file=output_file, max_workers=max_workers)
        analyzer.analyze_directory()
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"分析过程中发生错误: {e}")


if __name__ == "__main__":
    main()