#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量FCG提取脚本
基于IDA Pro headless模式批量提取PE文件的函数调用图(FCG)
支持benign和malware两类文件的分类处理
"""

import os
import sys
import json
import time
import logging
import hashlib
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple

class FCGBatchProcessor:
    """FCG批量处理器"""
    
    def __init__(self, ida_path: str, max_workers: int = 4, timeout: int = 300, verbose: bool = False):
        """
        初始化FCG批量处理器
        
        Args:
            ida_path: IDA Pro可执行文件路径
            max_workers: 最大并行工作线程数
            timeout: 单个文件处理超时时间（秒）
            verbose: 是否启用详细日志
        """
        self.ida_path = ida_path
        self.max_workers = max_workers
        self.timeout = timeout
        self.verbose = verbose
        
        # 设置日志
        self.setup_logging()
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'skipped_files': 0,
            'benign_processed': 0,
            'malware_processed': 0
        }
        
        # 进度文件路径
        self.progress_file = None
        
    def setup_logging(self):
        """设置日志配置"""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('fcg_batch_processing.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_file_hash(self, file_path: str) -> Optional[str]:
        """计算文件SHA256哈希值"""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256()
                while chunk := f.read(8192):
                    file_hash.update(chunk)
                return file_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return None
    
    def save_progress(self, progress_data: Dict):
        """保存处理进度"""
        if self.progress_file:
            try:
                with open(self.progress_file, 'w', encoding='utf-8') as f:
                    json.dump(progress_data, f, indent=2, ensure_ascii=False)
                self.logger.debug(f"Progress saved to {self.progress_file}")
            except Exception as e:
                self.logger.error(f"Error saving progress: {e}")
    
    def load_progress(self) -> Dict:
        """加载处理进度"""
        if self.progress_file and os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                self.logger.info(f"Progress loaded from {self.progress_file}")
                return progress_data
            except Exception as e:
                self.logger.error(f"Error loading progress: {e}")
        return {}
    
    def analyze_single_file(self, input_file: str, output_dir: str, file_type: str) -> Tuple[bool, str]:
        """
        分析单个PE文件
        
        Args:
            input_file: 输入PE文件路径
            output_dir: 输出目录
            file_type: 文件类型 ('benign' 或 'malware')
            
        Returns:
            (成功标志, 错误信息)
        """
        try:
            self.logger.info(f"Processing {file_type} file: {os.path.basename(input_file)}")
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取FCG脚本路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            ida_script = os.path.join(script_dir, 'ida_fcg_script.py')
            
            if not os.path.exists(ida_script):
                return False, f"IDA script not found: {ida_script}"
            
            # 设置环境变量
            env = os.environ.copy()
            env['IDA_OUTPUT_DIR'] = output_dir
            env['TVHEADLESS'] = '1'
            env['IDALOG'] = os.path.join(output_dir, f"ida_log_{int(time.time())}.txt")
            
            # 构建IDA命令
            ida_cmd = [
                self.ida_path,
                '-A',  # 自动分析
                '-S' + ida_script,  # 执行脚本
                '-L' + env['IDALOG'],  # 日志文件
                input_file
            ]
            
            self.logger.debug(f"IDA command: {' '.join(ida_cmd)}")
            
            # 执行IDA分析
            start_time = time.time()
            process = subprocess.Popen(
                ida_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=output_dir
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time
                
                if process.returncode == 0:
                    self.logger.info(f"Successfully processed {os.path.basename(input_file)} in {execution_time:.2f}s")
                    return True, ""
                else:
                    error_msg = f"IDA process failed with return code {process.returncode}"
                    if stderr:
                        error_msg += f": {stderr.decode('utf-8', errors='ignore')}"
                    self.logger.error(f"Failed to process {os.path.basename(input_file)}: {error_msg}")
                    return False, error_msg
                    
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                error_msg = f"Process timeout after {self.timeout} seconds"
                self.logger.error(f"Timeout processing {os.path.basename(input_file)}: {error_msg}")
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Exception during analysis: {str(e)}"
            self.logger.error(f"Error processing {os.path.basename(input_file)}: {error_msg}")
            return False, error_msg
    
    def get_pe_files(self, directory: str, extensions: List[str] = None) -> List[str]:
        """获取目录中的PE文件列表"""
        if extensions is None:
            extensions = ['.exe', '.dll', '.sys', '.ocx']
        
        pe_files = []
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    # 检查有后缀的文件
                    if any(file.lower().endswith(ext) for ext in extensions):
                        pe_files.append(file_path)
                    # 检查无后缀的文件
                    elif '.' not in file and self.is_pe_file(file_path):
                        pe_files.append(file_path)
            
            self.logger.info(f"Found {len(pe_files)} PE files in {directory}")
            return pe_files
            
        except Exception as e:
            self.logger.error(f"Error scanning directory {directory}: {e}")
            return []
    
    def is_pe_file(self, file_path: str) -> bool:
        """检查文件是否为PE文件"""
        try:
            with open(file_path, 'rb') as f:
                # 检查DOS头
                dos_header = f.read(2)
                if dos_header != b'MZ':
                    return False
                
                # 跳到PE头位置
                f.seek(60)
                pe_offset_bytes = f.read(4)
                if len(pe_offset_bytes) < 4:
                    return False
                
                pe_offset = int.from_bytes(pe_offset_bytes, byteorder='little')
                f.seek(pe_offset)
                
                # 检查PE签名
                pe_signature = f.read(4)
                return pe_signature == b'PE\x00\x00'
                
        except Exception:
            return False
    
    def process_file_batch(self, file_info_list: List[Tuple[str, str, str]]) -> Dict:
        """
        批量处理文件
        
        Args:
            file_info_list: [(input_file, output_dir, file_type), ...]
            
        Returns:
            处理结果统计
        """
        results = {
            'successful': [],
            'failed': [],
            'total': len(file_info_list)
        }
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {}
            for input_file, output_dir, file_type in file_info_list:
                future = executor.submit(self.analyze_single_file, input_file, output_dir, file_type)
                future_to_file[future] = (input_file, file_type)
            
            # 处理完成的任务
            completed = 0
            for future in as_completed(future_to_file):
                input_file, file_type = future_to_file[future]
                completed += 1
                
                try:
                    success, error_msg = future.result()
                    
                    if success:
                        results['successful'].append(input_file)
                        self.stats['successful_files'] += 1
                        if file_type == 'benign':
                            self.stats['benign_processed'] += 1
                        else:
                            self.stats['malware_processed'] += 1
                    else:
                        results['failed'].append((input_file, error_msg))
                        self.stats['failed_files'] += 1
                    
                    self.stats['processed_files'] += 1
                    
                    # 定期保存进度
                    if completed % 10 == 0:
                        progress_data = {
                            'stats': self.stats,
                            'timestamp': time.time(),
                            'completed_files': completed,
                            'total_files': results['total']
                        }
                        self.save_progress(progress_data)
                    
                    # 显示进度
                    progress = (completed / results['total']) * 100
                    self.logger.info(f"Progress: {completed}/{results['total']} ({progress:.1f}%)")
                    
                except Exception as e:
                    self.logger.error(f"Error processing {input_file}: {e}")
                    results['failed'].append((input_file, str(e)))
                    self.stats['failed_files'] += 1
        
        return results
    
    def batch_analyze(self, input_configs: List[Dict], extensions: List[str] = None) -> Dict:
        """
        批量分析指定配置的PE文件
        
        Args:
            input_configs: 输入输出配置列表，每个配置包含:
                          {'input_dir': str, 'output_dir': str, 'file_type': str}
            extensions: 支持的文件扩展名列表
            
        Returns:
            处理结果统计
        """
        self.logger.info("Starting batch FCG analysis...")
        
        # 设置进度文件
        self.progress_file = os.path.join(os.getcwd(), 'fcg_batch_progress.json')
        
        # 加载之前的进度
        progress_data = self.load_progress()
        processed_files = set(progress_data.get('processed_files', []))
        
        # 收集所有需要处理的文件
        file_info_list = []
        
        # 处理每个输入配置
        for config in input_configs:
            input_dir = config['input_dir']
            output_dir = config['output_dir']
            file_type = config['file_type']
            
            if os.path.exists(input_dir):
                files = self.get_pe_files(input_dir, extensions)
                new_files = [f for f in files if f not in processed_files]
                
                for file_path in new_files:
                    file_info_list.append((file_path, output_dir, file_type))
                
                self.logger.info(f"Found {len(files)} {file_type} files, {len(new_files)} new")
            else:
                self.logger.warning(f"{file_type.capitalize()} directory not found: {input_dir}")
        
        self.stats['total_files'] = len(file_info_list)
        
        if not file_info_list:
            self.logger.info("No new files to process")
            return {'message': 'No new files to process'}
        
        self.logger.info(f"Total files to process: {len(file_info_list)}")
        
        # 确保所有输出目录存在
        for config in input_configs:
            os.makedirs(config['output_dir'], exist_ok=True)
        
        # 开始批量处理
        start_time = time.time()
        results = self.process_file_batch(file_info_list)
        total_time = time.time() - start_time
        
        # 最终统计
        self.logger.info("="*60)
        self.logger.info("FCG Batch Analysis Completed!")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info(f"Total files: {self.stats['total_files']}")
        self.logger.info(f"Successful: {self.stats['successful_files']}")
        self.logger.info(f"Failed: {self.stats['failed_files']}")
        self.logger.info(f"Benign processed: {self.stats['benign_processed']}")
        self.logger.info(f"Malware processed: {self.stats['malware_processed']}")
        
        if self.stats['failed_files'] > 0:
            self.logger.info("Failed files:")
            for failed_file, error in results['failed']:
                self.logger.info(f"  - {os.path.basename(failed_file)}: {error}")
        
        # 保存最终进度
        final_progress = {
            'stats': self.stats,
            'timestamp': time.time(),
            'completed': True,
            'total_time': total_time,
            'processed_files': list(processed_files) + [f[0] for f in file_info_list]
        }
        self.save_progress(final_progress)
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量FCG提取工具')
    
    # 输入目录参数 - 至少需要一个
    parser.add_argument('--benign-dir', help='Benign文件目录')
    parser.add_argument('--malware-dir', help='Malware文件目录')
    parser.add_argument('--input-dir', help='通用输入目录（当只处理一种类型文件时使用）')
    
    # 输出目录参数 - 对应输入目录
    parser.add_argument('--benign-output', help='Benign输出目录')
    parser.add_argument('--malware-output', help='Malware输出目录')
    parser.add_argument('--output-dir', help='通用输出目录（当只处理一种类型文件时使用）')
    
    # 文件类型参数 - 当使用通用目录时指定文件类型
    parser.add_argument('--file-type', choices=['benign', 'malware'], default='benign',
                       help='文件类型（当使用--input-dir时需要指定，默认: benign）')
    parser.add_argument('--ida-path', default=r"D:\download\IDA Professional 9.1\ida.exe", 
                       help='IDA Pro可执行文件路径（默认: D:\\download\\IDA Professional 9.1\\ida.exe）')
    parser.add_argument('--max-workers', type=int, default=1, 
                       help='最大并行工作线程数（默认: 1，建议设为1避免IDA Pro资源冲突）')
    parser.add_argument('--timeout', type=int, default=300, help='单个文件处理超时时间（秒）')
    parser.add_argument('--extensions', nargs='+', default=['.exe', '.dll', '.sys', '.ocx'], 
                       help='支持的文件扩展名')
    parser.add_argument('--verbose', action='store_true', help='启用详细日志')
    
    args = parser.parse_args()
    
    # 参数验证逻辑
    # 检查输入目录参数
    input_configs = []
    
    # 方式1: 使用通用输入输出目录
    if args.input_dir and args.output_dir:
        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory not found: {args.input_dir}")
            sys.exit(1)
        input_configs.append({
            'input_dir': args.input_dir,
            'output_dir': args.output_dir,
            'file_type': args.file_type
        })
    
    # 方式2: 使用分别的benign/malware目录
    if args.benign_dir and args.benign_output:
        if not os.path.exists(args.benign_dir):
            print(f"Error: Benign directory not found: {args.benign_dir}")
            sys.exit(1)
        input_configs.append({
            'input_dir': args.benign_dir,
            'output_dir': args.benign_output,
            'file_type': 'benign'
        })
    
    if args.malware_dir and args.malware_output:
        if not os.path.exists(args.malware_dir):
            print(f"Error: Malware directory not found: {args.malware_dir}")
            sys.exit(1)
        input_configs.append({
            'input_dir': args.malware_dir,
            'output_dir': args.malware_output,
            'file_type': 'malware'
        })
    
    # 检查是否至少有一个有效配置
    if not input_configs:
        print("Error: No valid input-output configuration found!")
        print("Please provide one of the following:")
        print("  1. --input-dir and --output-dir (with optional --file-type)")
        print("  2. --benign-dir and --benign-output")
        print("  3. --malware-dir and --malware-output")
        print("  4. Both benign and malware configurations")
        sys.exit(1)
    
    print(f"Found {len(input_configs)} input-output configuration(s):")
    for i, config in enumerate(input_configs, 1):
        print(f"  {i}. {config['file_type']}: {config['input_dir']} -> {config['output_dir']}")
    
    # 验证IDA路径
    if not os.path.exists(args.ida_path):
        print(f"Error: IDA Pro executable not found: {args.ida_path}")
        print(f"Please check the path or specify correct path using --ida-path argument")
        print(f"Common IDA Pro installation paths:")
        print(f"  - D:\\download\\IDA Professional 9.1\\ida.exe")
        print(f"  - C:\\Program Files\\IDA Professional\\ida.exe")
        print(f"  - C:\\Program Files (x86)\\IDA Professional\\ida.exe")
        sys.exit(1)
    
    # 多线程安全警告
    if args.max_workers > 1:
        print(f"Warning: Using {args.max_workers} workers may cause IDA Pro resource conflicts.")
        print(f"For stable operation, consider using --max-workers 1")
        response = input("Continue with multiple workers? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled. Use --max-workers 1 for safe operation.")
            sys.exit(0)
    
    # 创建处理器
    processor = FCGBatchProcessor(
        ida_path=args.ida_path,
        max_workers=args.max_workers,
        timeout=args.timeout,
        verbose=args.verbose
    )
    
    try:
        # 开始批量分析
        results = processor.batch_analyze(
            input_configs=input_configs,
            extensions=args.extensions
        )
        
        print("\nBatch FCG analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during batch analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()