#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FCG特征L2归一化脚本
用于对指定路径下的.pt文件进行L2归一化处理
支持批量处理恶意软件和良性软件的FCG特征文件
"""

import os
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import argparse
from tqdm import tqdm
import concurrent.futures
from torch_geometric.data import Data

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def l2_normalize_features(data: Data) -> Data:
    """
    对PyTorch Geometric Data对象中的特征矩阵进行L2归一化
    
    Args:
        data: PyTorch Geometric Data对象，包含x (特征矩阵) 和 edge_index
        
    Returns:
        归一化后的Data对象
    """
    if data.x is None or data.x.size(0) == 0:
        logger.warning("特征矩阵为空，跳过归一化")
        return data
    
    # 对特征矩阵进行L2归一化 (按行归一化)
    # F.normalize默认在最后一个维度上进行归一化，p=2表示L2范数
    normalized_x = F.normalize(data.x, p=2, dim=1)
    
    # 创建新的Data对象，保持其他属性不变
    normalized_data = Data(
        x=normalized_x,
        edge_index=data.edge_index,
        edge_attr=getattr(data, 'edge_attr', None),
        y=getattr(data, 'y', None),
        pos=getattr(data, 'pos', None)
    )
    
    return normalized_data


def process_single_file(file_path: Path, output_dir: Optional[Path] = None, 
                       backup: bool = True) -> Tuple[bool, str]:
    """
    处理单个.pt文件进行L2归一化
    
    Args:
        file_path: 输入.pt文件路径
        output_dir: 输出目录，如果为None则覆盖原文件
        backup: 是否备份原文件
        
    Returns:
        (成功标志, 错误信息)
    """
    try:
        # 加载.pt文件
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        
        # 检查是否为PyTorch Geometric Data对象
        if not isinstance(data, Data):
            return False, f"文件 {file_path.name} 不是有效的PyTorch Geometric Data对象"
        
        # 检查特征矩阵
        if data.x is None:
            return False, f"文件 {file_path.name} 缺少特征矩阵 x"
        
        original_shape = data.x.shape
        original_norm = torch.norm(data.x, p=2, dim=1).mean().item()
        
        # 进行L2归一化
        normalized_data = l2_normalize_features(data)
        
        # 验证归一化结果
        normalized_norm = torch.norm(normalized_data.x, p=2, dim=1).mean().item()
        
        # 确定输出路径
        if output_dir is not None:
            output_path = output_dir / file_path.name
        else:
            output_path = file_path
            
        # 备份原文件（如果需要且是原地更新）
        if backup and output_dir is None:
            backup_path = file_path.with_suffix('.pt.backup')
            if not backup_path.exists():
                torch.save(data, backup_path)
                logger.debug(f"备份原文件到: {backup_path}")
        
        # 保存归一化后的数据
        torch.save(normalized_data, output_path)
        
        logger.info(f"成功处理 {file_path.name}: 形状={original_shape}, "
                   f"原始L2范数均值={original_norm:.4f}, 归一化后L2范数均值={normalized_norm:.4f}")
        
        return True, ""
        
    except Exception as e:
        error_msg = f"处理文件 {file_path.name} 时出错: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def batch_normalize_directory(input_dir: str, output_dir: Optional[str] = None,
                            backup: bool = True, num_workers: int = 4,
                            pattern: str = "*.pt") -> Tuple[int, int, List[str]]:
    """
    批量处理目录下的.pt文件进行L2归一化
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为None则覆盖原文件
        backup: 是否备份原文件
        num_workers: 并行处理的工作进程数
        pattern: 文件匹配模式
        
    Returns:
        (成功数量, 失败数量, 错误信息列表)
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")
    
    # 创建输出目录
    output_path = None
    if output_dir is not None and isinstance(output_dir, (str, Path)):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {output_path}")
    else:
        logger.info("将覆盖原文件")
    
    # 查找所有.pt文件
    pt_files = list(input_path.glob(pattern))
    if not pt_files:
        logger.warning(f"在 {input_dir} 中未找到匹配 {pattern} 的文件")
        return 0, 0, []
    
    logger.info(f"找到 {len(pt_files)} 个文件，开始批量处理...")
    
    success_count = 0
    failed_count = 0
    error_messages = []
    
    # 使用进度条和多进程处理
    with tqdm(total=len(pt_files), desc="处理进度") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(process_single_file, file_path, output_path, backup): file_path
                for file_path in pt_files
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    success, error_msg = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        error_messages.append(error_msg)
                except Exception as e:
                    failed_count += 1
                    error_msg = f"处理文件 {file_path.name} 时发生异常: {str(e)}"
                    error_messages.append(error_msg)
                    logger.error(error_msg)
                
                pbar.update(1)
    
    logger.info(f"批量处理完成: 成功 {success_count} 个，失败 {failed_count} 个")
    
    return success_count, failed_count, error_messages


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FCG特征L2归一化工具')
    parser.add_argument('--input_dirs', nargs='+', required=True,
                       help='输入目录路径列表')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录路径（可选，默认覆盖原文件）')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='是否备份原文件（仅在覆盖模式下有效）')
    parser.add_argument('--no-backup', dest='backup', action='store_false',
                       help='不备份原文件')
    parser.add_argument('--workers', type=int, default=4,
                       help='并行处理的工作进程数（默认4）')
    parser.add_argument('--pattern', type=str, default='*.pt',
                       help='文件匹配模式（默认*.pt）')
    
    args = parser.parse_args()
    
    total_success = 0
    total_failed = 0
    all_errors = []
    
    # 处理每个输入目录
    for input_dir in args.input_dirs:
        logger.info(f"开始处理目录: {input_dir}")
        
        try:
            success, failed, errors = batch_normalize_directory(
                input_dir=input_dir,
                output_dir=args.output_dir,
                backup=args.backup,
                num_workers=args.workers,
                pattern=args.pattern
            )
            
            total_success += success
            total_failed += failed
            all_errors.extend(errors)
            
        except Exception as e:
            logger.error(f"处理目录 {input_dir} 时出错: {str(e)}")
            total_failed += 1
            all_errors.append(f"目录 {input_dir}: {str(e)}")
    
    # 输出最终统计
    logger.info(f"所有目录处理完成:")
    logger.info(f"  总成功: {total_success} 个文件")
    logger.info(f"  总失败: {total_failed} 个文件")
    
    if all_errors:
        logger.info("错误详情:")
        for error in all_errors[:10]:  # 只显示前10个错误
            logger.error(f"  {error}")
        if len(all_errors) > 10:
            logger.info(f"  ... 还有 {len(all_errors) - 10} 个错误")


if __name__ == "__main__":
    # 如果直接运行，使用默认配置处理指定路径
    if len(os.sys.argv) == 1:
        # 默认配置
        input_directories = [
            "/mnt/lbq/dataset/malware_fcg",
            "/mnt/lbq/dataset/benign_fcg"
        ]
        
        logger.info("使用默认配置运行...")
        logger.info(f"输入目录: {input_directories}")
        
        total_success = 0
        total_failed = 0
        
        for input_dir in input_directories:
            if os.path.exists(input_dir):
                logger.info(f"开始处理目录: {input_dir}")
                success, failed, errors = batch_normalize_directory(
                    input_dir=input_dir,
                    output_dir=None,  # 覆盖原文件
                    backup=False,     # 不备份原文件
                    num_workers=4,    # 4个工作进程
                    pattern="*.pt"    # 处理所有.pt文件
                )
                total_success += success
                total_failed += failed
                
                if errors:
                    logger.warning(f"目录 {input_dir} 处理中的错误:")
                    for error in errors[:5]:  # 只显示前5个错误
                        logger.error(f"  {error}")
            else:
                logger.warning(f"目录不存在，跳过: {input_dir}")
        
        logger.info(f"默认配置处理完成: 成功 {total_success} 个，失败 {total_failed} 个")
    else:
        # 使用命令行参数
        main()