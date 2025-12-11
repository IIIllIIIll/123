#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件重命名脚本
功能：将指定文件夹内的所有文件重命名为SHA256值
支持无后缀文件和.exe文件等各种格式
"""

import os
import hashlib
import re
import signal
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class FileRenamer:
    def __init__(self, target_directory, max_workers=8):
        """
        初始化文件重命名器
        
        Args:
            target_directory (str): 目标文件夹路径
            max_workers (int): 最大线程数
        """
        self.target_directory = Path(target_directory)
        self.max_workers = max_workers
        self.renamed_count = 0
        self.duplicate_count = 0
        self.failed_count = 0
        self.interrupted = False  # 中断标志
        self.processed_sha256s = set()  # 记录已处理的SHA256值，避免重复处理
        
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
    
    def calculate_sha256(self, file_path):
        """
        计算文件的SHA256哈希值
        
        Args:
            file_path (Path): 文件路径
            
        Returns:
            str: SHA256哈希值，如果计算失败返回None
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # 分块读取文件以处理大文件
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"计算SHA256失败 {file_path.name}: {e}")
            return None
    
    def _get_full_extension(self, filename):
        """
        获取文件的完整扩展名，支持多重扩展名
        
        Args:
            filename (str): 文件名
            
        Returns:
            str: 完整的扩展名
        """
        # 常见的多重扩展名模式
        multi_extensions = ['.tar.gz', '.tar.bz2', '.tar.xz', '.tar.Z']
        
        filename_lower = filename.lower()
        for ext in multi_extensions:
            if filename_lower.endswith(ext):
                return ext
        
        # 如果不是多重扩展名，返回普通扩展名
        return Path(filename).suffix
    
    def is_sha256_filename(self, filename):
        """
        检查文件名是否为SHA256格式
        
        Args:
            filename (str): 文件名
            
        Returns:
            bool: 是否为SHA256格式的文件名
        """
        # SHA256是64位十六进制字符串，可能带有扩展名
        name_without_ext = Path(filename).stem
        return bool(re.match(r'^[a-fA-F0-9]{64}$', name_without_ext))
    
    def rename_to_sha256(self, file_path):
        """
        将文件重命名为SHA256值
        
        Args:
            file_path (Path): 原文件路径
            
        Returns:
            tuple: (新文件路径, 是否为重复文件)，如果重命名失败返回(None, False)
        """
        sha256_value = self.calculate_sha256(file_path)
        if not sha256_value:
            return None, False
        
        # 检查是否已经处理过这个SHA256值
        with self.lock:
            if sha256_value in self.processed_sha256s:
                print(f"SHA256值已存在: {sha256_value}，跳过文件: {file_path.name}")
                self.duplicate_count += 1
                return None, True
            
            # 添加到已处理集合
            self.processed_sha256s.add(sha256_value)
        
        # 获取完整的文件扩展名（支持多重扩展名如.tar.gz）
        extension = self._get_full_extension(file_path.name)
        new_filename = f"{sha256_value}{extension}"
        new_file_path = file_path.parent / new_filename
        
        # 检查目标文件是否已存在
        if new_file_path.exists():
            print(f"目标文件已存在: {new_filename}，跳过文件: {file_path.name}")
            with self.lock:
                self.duplicate_count += 1
            return None, True
        
        # 尝试重命名文件
        try:
            file_path.rename(new_file_path)
            with self.lock:
                self.renamed_count += 1
            print(f"重命名成功: {file_path.name} -> {new_filename}")
            return new_file_path, False
        except FileExistsError:
            # 如果重命名时发现文件已存在（可能是其他线程刚创建的）
            print(f"重命名时发现目标文件已存在: {new_filename}，跳过文件: {file_path.name}")
            with self.lock:
                self.duplicate_count += 1
            return None, True
        except Exception as e:
            # 其他可能导致重命名失败的异常
            print(f"重命名失败 {file_path.name}: {e}")
            with self.lock:
                self.failed_count += 1
            return None, False
    
    def process_single_file(self, file_path):
        """
        处理单个文件
        
        Args:
            file_path (Path): 文件路径
            
        Returns:
            bool: 处理是否成功
        """
        # 检查是否被中断
        if self.interrupted:
            return False
            
        original_name = file_path.name
        
        # 重命名文件
        renamed_path, is_duplicate = self.rename_to_sha256(file_path)
        
        # 如果是重复文件，直接返回成功
        if is_duplicate:
            print(f"重复文件处理完成: {original_name}")
            return True
        
        # 如果重命名失败
        if not renamed_path:
            print(f"文件重命名失败: {original_name}")
            return False
        
        print(f"文件重命名成功: {original_name} -> {renamed_path.name}")
        return True
    
    def process_directory(self):
        """
        处理整个目录（多线程版本）
        """
        print(f"开始处理目录: {self.target_directory}")
        print(f"使用线程数: {self.max_workers}")
        print("-" * 50)
        
        # 预扫描目录，识别已存在的SHA256文件
        print("正在预扫描目录，识别已存在的SHA256文件...")
        existing_files = list(self.target_directory.iterdir())
        for file_path in existing_files:
            if file_path.is_file() and self.is_sha256_filename(file_path.name):
                # 提取SHA256值（去掉扩展名）
                sha256_part = file_path.stem
                if len(sha256_part) == 64:  # SHA256长度为64个字符
                    self.processed_sha256s.add(sha256_part)
        
        print(f"发现 {len(self.processed_sha256s)} 个已存在的SHA256文件")
        
        # 获取所有需要处理的文件（排除已经是SHA256格式的文件）
        files = []
        for f in self.target_directory.iterdir():
            if f.is_file():
                # 跳过已经是SHA256格式的文件
                if self.is_sha256_filename(f.name):
                    continue
                files.append(f)
        
        total_files = len(files)
        
        if total_files == 0:
            print("目录中没有找到需要处理的文件")
            return
        
        print(f"找到 {total_files} 个需要处理的文件")
        
        # 使用多线程处理文件
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_file = {executor.submit(self.process_single_file, file_path): file_path 
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
                        print(f"[{completed}/{total_files}] 完成处理: {file_path.name}")
                    except Exception as exc:
                        print(f"[{completed}/{total_files}] 处理异常 {file_path.name}: {exc}")
        except KeyboardInterrupt:
            print("\n用户中断操作，正在停止处理...")
            self.interrupted = True
            return
        
        # 输出统计信息
        self.print_summary()
    
    def print_summary(self):
        """
        打印处理结果摘要
        """
        print("\n" + "=" * 60)
        print("文件重命名完成！统计信息：")
        print(f"成功重命名文件数: {self.renamed_count}")
        print(f"重复文件数: {self.duplicate_count}")
        print(f"失败文件数: {self.failed_count}")
        print("=" * 60)


def main():
    """
    主函数：交互式运行或直接配置运行
    """
    # 交互式输入
    try:
        target_directory = input("请输入要处理的目录路径: ").strip()
        if not target_directory:
            # 如果没有输入，使用默认路径
            target_directory = r"D:\Test\AAAAAAAAAAAAAAA\demooooooo\dataset\malware"
            print(f"使用默认目录: {target_directory}")
        
        max_workers_input = input("请输入线程数 (默认8): ").strip()
        if max_workers_input and max_workers_input.isdigit():
            max_workers = int(max_workers_input)
        else:
            max_workers = 8
            print(f"使用默认线程数: {max_workers}")
        
        renamer = FileRenamer(target_directory, max_workers=max_workers)
        renamer.process_directory()
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == "__main__":
    main()