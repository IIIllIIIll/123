#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IDA Pro批量处理脚本 - Headless模式
使用命令行方式批量调用IDA Pro进行分析
"""

import os
import sys
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_ida_headless.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IDAHeadlessProcessor:
    """IDA Pro Headless模式批量处理器"""
    
    def __init__(self, ida_path: str = r"D:\download\IDA Professional 9.1\ida.exe"):
        self.ida_path = ida_path
        self.ida_script_path = None
        self.processed_files = set()
        self.failed_files = set()
        
        # 验证IDA Pro安装
        if not os.path.exists(self.ida_path):
            logger.error(f"IDA Pro未找到: {self.ida_path}")
            raise FileNotFoundError(f"IDA Pro未找到: {self.ida_path}")
        
        # 设置IDA脚本路径
        current_dir = Path(__file__).parent
        self.ida_script_path = current_dir / "ida_script.py"
        
        if not self.ida_script_path.exists():
            logger.error(f"IDA脚本未找到: {self.ida_script_path}")
            raise FileNotFoundError(f"IDA脚本未找到: {self.ida_script_path}")
        
        logger.info(f"使用IDA Pro: {self.ida_path}")
        logger.info(f"使用IDA脚本: {self.ida_script_path}")
    
    def load_progress(self, progress_file: str):
        """加载处理进度"""
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    self.processed_files = set(progress_data.get('processed', []))
                    self.failed_files = set(progress_data.get('failed', []))
                    logger.info(f"已加载进度: 已处理 {len(self.processed_files)} 个文件，失败 {len(self.failed_files)} 个文件")
            except Exception as e:
                logger.error(f"加载进度文件失败: {e}")
    
    def save_progress(self, progress_file: str):
        """保存处理进度"""
        try:
            progress_data = {
                'processed': list(self.processed_files),
                'failed': list(self.failed_files),
                'timestamp': time.time()
            }
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存进度文件失败: {e}")
    
    def analyze_single_file(self, file_path: str, output_dir: str, timeout: int = 300) -> bool:
        """分析单个文件"""
        try:
            file_path = Path(file_path).resolve()
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 检查是否已处理
            file_key = str(file_path)
            if file_key in self.processed_files:
                logger.info(f"文件已处理，跳过: {file_path.name}")
                return True
            
            if file_key in self.failed_files:
                logger.info(f"文件之前失败，跳过: {file_path.name}")
                return False
            
            # 检查输出文件是否已存在
            expected_output = output_dir / f"{file_path.stem}_ida_analysis.json"
            if expected_output.exists():
                logger.info(f"输出文件已存在，跳过: {file_path.name}")
                self.processed_files.add(file_key)
                return True
            
            logger.info(f"开始分析文件: {file_path.name}")
            
            # 设置环境变量，传递输出目录给IDA脚本
            env = os.environ.copy()
            env['IDA_OUTPUT_DIR'] = str(output_dir)
            
            # 构建IDA命令
            cmd = [
                str(self.ida_path),
                '-A',  # 自动模式（无GUI）
                f'-S{self.ida_script_path}',  # 执行脚本（注意-S和路径之间没有空格）
                str(file_path)  # 要分析的文件
            ]
            
            logger.debug(f"执行命令: {' '.join(cmd)}")
            
            # 执行IDA分析
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(output_dir)
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                end_time = time.time()
                
                # 检查结果
                if process.returncode == 0 and expected_output.exists():
                    logger.info(f"文件 {file_path.name} 分析完成 (耗时: {end_time - start_time:.1f}秒)")
                    self.processed_files.add(file_key)
                    
                    # 记录输出信息（如果有的话）
                    if stdout.strip():
                        logger.debug(f"IDA输出: {stdout.strip()}")
                    
                    return True
                else:
                    logger.error(f"IDA分析失败: {file_path.name}, 返回码: {process.returncode}")
                    if stderr:
                        logger.error(f"错误输出: {stderr[:500]}")
                    if stdout:
                        logger.debug(f"标准输出: {stdout[:500]}")
                    
                    self.failed_files.add(file_key)
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.error(f"IDA分析超时 ({timeout}秒): {file_path.name}")
                process.kill()
                process.wait()
                self.failed_files.add(file_key)
                return False
                
        except Exception as e:
            logger.error(f"分析文件 {file_path} 时出错: {e}")
            self.failed_files.add(str(file_path))
            return False
    
    def is_pe_file(self, file_path: Path) -> bool:
        """检查文件是否为PE文件（优化版本）"""
        try:
            # 检查文件大小，PE文件至少需要几百字节
            file_size = file_path.stat().st_size
            if file_size < 64:  # PE头至少需要64字节
                return False
            
            # 限制检查的文件大小，避免处理过大的文件
            if file_size > 100 * 1024 * 1024:  # 100MB
                logger.warning(f"文件过大，跳过PE检查: {file_path.name} ({file_size / 1024 / 1024:.1f}MB)")
                return False
            
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
                
                # 验证PE偏移是否合理
                if pe_offset < 64 or pe_offset > file_size - 4:
                    return False
                
                f.seek(pe_offset)
                
                # 检查PE签名
                pe_signature = f.read(4)
                return pe_signature == b'PE\x00\x00'
                
        except Exception:
            return False
    
    def get_pe_files(self, input_path: Path) -> List[Path]:
        """获取目录中的PE文件，过滤掉IDA临时文件"""
        pe_files = []
        ida_extensions = {'.i64', '.id0', '.id1', '.id2', '.nam', '.til', '.idb'}
        standard_pe_extensions = {'.exe', '.dll', '.sys', '.ocx'}
        
        # 获取所有文件
        all_files = [f for f in input_path.iterdir() if f.is_file()]
        total_files = len(all_files)
        
        logger.info(f"开始扫描目录: {input_path}")
        logger.info(f"总文件数: {total_files}")
        
        processed_count = 0
        pe_count = 0
        
        for file_path in all_files:
            processed_count += 1
            
            # 跳过IDA临时文件
            if file_path.suffix.lower() in ida_extensions:
                continue
                
            # 检查PE文件
            is_pe = False
            if file_path.suffix.lower() in standard_pe_extensions:
                is_pe = True
                pe_files.append(file_path)
                pe_count += 1
            elif file_path.suffix.lower() not in standard_pe_extensions:
                # 对于没有标准PE后缀的文件，检查是否为PE文件
                if self.is_pe_file(file_path):
                    is_pe = True
                    pe_files.append(file_path)
                    pe_count += 1
            
            # 显示进度（每100个文件或最后一个文件）
            if processed_count % 100 == 0 or processed_count == total_files:
                logger.info(f"扫描进度: {processed_count}/{total_files} ({processed_count/total_files*100:.1f}%) - 已找到PE文件: {pe_count}")
                    
        logger.info(f"扫描完成! 总共找到 {pe_count} 个PE文件")
        return pe_files
    def batch_analyze(self, input_dir: str, output_dir: str, max_workers: int = 1, timeout: int = 300, file_extensions: List[str] = None):
        """批量分析文件"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        if not input_path.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            return
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 设置进度文件
        progress_file = output_path / "processing_progress.json"
        self.load_progress(str(progress_file))
        
        # 获取所有要处理的文件
        if file_extensions is None:
            # 使用新的PE文件过滤方法
            all_files = self.get_pe_files(input_path)
        else:
            all_files = []
            for ext in file_extensions:
                all_files.extend(list(input_path.glob(ext)))
        
        # 过滤已处理的文件
        remaining_files = [f for f in all_files if str(f) not in self.processed_files and f.is_file()]
        
        logger.info(f"找到 {len(all_files)} 个文件，需要处理 {len(remaining_files)} 个文件")
        
        if not remaining_files:
            logger.info("所有文件已处理完成")
            return
        
        success_count = len(self.processed_files)
        total_count = len(all_files)
        
        # 使用线程池进行并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_file = {
                executor.submit(self.analyze_single_file, str(pe_file), str(output_path), timeout): pe_file
                for pe_file in remaining_files
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_file):
                pe_file = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                    
                    # 定期保存进度
                    if (len(self.processed_files) + len(self.failed_files)) % 10 == 0:
                        self.save_progress(str(progress_file))
                        
                    logger.info(f"进度: {len(self.processed_files) + len(self.failed_files)}/{total_count} "
                              f"(成功: {len(self.processed_files)}, 失败: {len(self.failed_files)})")
                    
                except Exception as e:
                    logger.error(f"处理文件 {pe_file} 时出错: {e}")
                    self.failed_files.add(str(pe_file))
        
        # 保存最终进度
        self.save_progress(str(progress_file))
        
        logger.info(f"批量分析完成!")
        logger.info(f"总文件数: {total_count}")
        logger.info(f"成功分析: {len(self.processed_files)}")
        logger.info(f"分析失败: {len(self.failed_files)}")
        
        if self.failed_files:
            logger.info("失败的文件:")
            for failed_file in list(self.failed_files)[:10]:  # 只显示前10个
                logger.info(f"  - {Path(failed_file).name}")
            if len(self.failed_files) > 10:
                logger.info(f"  ... 还有 {len(self.failed_files) - 10} 个失败文件")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='IDA Pro批量处理工具 - Headless模式')
    parser.add_argument('--input', '-i', required=True, help='输入目录路径')
    parser.add_argument('--output', '-o', required=True, help='输出目录路径')
    parser.add_argument('--ida-path', default=r"D:\download\IDA Professional 9.1\ida.exe", help='IDA Pro路径')
    parser.add_argument('--max-workers', type=int, default=1, help='最大并发数（建议设为1避免资源冲突）')
    parser.add_argument('--timeout', type=int, default=300, help='单个文件分析超时时间（秒）')
    parser.add_argument('--extensions', nargs='+', default=None, help='要处理的文件扩展名（默认处理所有文件）')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        processor = IDAHeadlessProcessor(args.ida_path)
        processor.batch_analyze(
            input_dir=args.input,
            output_dir=args.output,
            max_workers=args.max_workers,
            timeout=args.timeout,
            file_extensions=args.extensions
        )
            
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()