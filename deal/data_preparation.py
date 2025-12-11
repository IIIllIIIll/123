#!/usr/bin/env python3
"""
总的数据准备与组织阶段
划分训练集、测试集、验证集
创建标签与数据划分脚本
"""

import os
import csv
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPreparation:
    """数据准备与组织类"""
    
    def __init__(self,
                 benign_dir: str = "D:/Test/AAAAAAAAAAAAAAA/demooooooo/dataset/benign",
                 malware_dir: str = "D:/Test/AAAAAAAAAAAAAAA/demooooooo/dataset/malware",
                 target_dir: str = "D:/Test/AAAAAAAAAAAAAAA/demooooooo/dataset/all_data",
                 output_dir: str = "D:/Test/AAAAAAAAAAAAAAA/demooooooo/output"):
        """
        初始化数据准备器
        
        Args:
            benign_dir: 良性样本目录路径 (a文件夹)
            malware_dir: 恶意样本目录路径 (b文件夹)
            target_dir: 目标数据目录 (指定目录)
            output_dir: 输出目录 (CSV文件保存位置)
        """
        self.benign_dir = benign_dir
        self.malware_dir = malware_dir
        self.target_dir = target_dir
        self.output_dir = output_dir
        
        # 数据集划分比例
        self.train_ratio = 0.7
        self.test_ratio = 0.2
        self.val_ratio = 0.1
        
    def scan_samples(self, directory: str, label: int) -> List[Dict]:
        """
        扫描指定目录下的所有样本文件
        
        Args:
            directory: 目录路径
            label: 标签 (0为良性，1为恶意)
            
        Returns:
            样本信息列表
        """
        samples = []
        
        if not directory or not os.path.exists(directory):
            logger.warning(f"目录不存在或为空: {directory}")
            return samples
        
        dir_path = Path(directory)
        
        # 扫描所有文件 (支持常见的可执行文件格式)
        file_extensions = ['.exe', '']  # 空字符串''表示无后缀文件
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                # 检查文件扩展名或者无扩展名的文件
                if (file_path.suffix.lower() in file_extensions or 
                    file_path.suffix == '' and file_path.name != ''):
                    
                    sample_info = {
                        'sample_id': file_path.stem,  # 文件名(不含扩展名)作为sample_id
                        'file_path': str(file_path),
                        'file_name': file_path.name,
                        'label': label
                    }
                    samples.append(sample_info)
        
        logger.info(f"在 {directory} 中找到 {len(samples)} 个样本文件")
        return samples
    
    def assign_splits(self, samples: List[Dict]) -> List[Dict]:
        """
        为样本分配数据集划分 (train/test/val)
        
        Args:
            samples: 样本信息列表
            
        Returns:
            包含split信息的样本列表
        """
        # 随机打乱样本顺序
        random.shuffle(samples)
        
        total_count = len(samples)
        train_count = int(total_count * self.train_ratio)
        test_count = int(total_count * self.test_ratio)
        val_count = total_count - train_count - test_count
        
        # 分配split标签
        for i, sample in enumerate(samples):
            if i < train_count:
                sample['split'] = 'train'
            elif i < train_count + test_count:
                sample['split'] = 'test'
            else:
                sample['split'] = 'val'
        
        logger.info(f"数据集划分: train={train_count}, test={test_count}, val={val_count}")
        return samples
    
    def create_labels_csv(self, all_samples: List[Dict]) -> str:
        """
        创建labels.csv文件
        
        Args:
            all_samples: 所有样本信息列表
            
        Returns:
            CSV文件路径
        """
        # 确保输出目录存在
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_file_path = output_path / 'labels.csv'
        
        # 写入CSV文件
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['sample_id', 'label', 'split']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 写入数据
            for sample in all_samples:
                writer.writerow({
                    'sample_id': sample['sample_id'],
                    'label': sample['label'],
                    'split': sample['split']
                })
        
        logger.info(f"成功创建labels.csv文件: {csv_file_path}")
        logger.info(f"总计 {len(all_samples)} 个样本")
        
        # 统计信息
        stats = {'train': 0, 'test': 0, 'val': 0, 'benign': 0, 'malware': 0}
        for sample in all_samples:
            stats[sample['split']] += 1
            if sample['label'] == 0:
                stats['benign'] += 1
            else:
                stats['malware'] += 1
        
        logger.info(f"统计信息: {stats}")
        
        return str(csv_file_path)
    
    def copy_samples_to_target(self, all_samples: List[Dict]) -> Dict[str, int]:
        """
        将所有样本复制到指定目录
        
        Args:
            all_samples: 所有样本信息列表
            
        Returns:
            复制统计信息
        """
        # 确保目标目录存在
        target_path = Path(self.target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        failed_count = 0
        
        for sample in all_samples:
            try:
                source_file = Path(sample['file_path'])
                target_file = target_path / sample['file_name']
                
                # 如果目标文件已存在，添加后缀避免冲突
                counter = 1
                original_target = target_file
                while target_file.exists():
                    stem = original_target.stem
                    suffix = original_target.suffix
                    target_file = target_path / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                # 复制文件
                shutil.copy2(source_file, target_file)
                success_count += 1
                
                logger.debug(f"复制成功: {source_file} -> {target_file}")
                
            except Exception as e:
                logger.error(f"复制失败 {sample['file_path']}: {e}")
                failed_count += 1
        
        stats = {
            'total': len(all_samples),
            'success': success_count,
            'failed': failed_count
        }
        
        logger.info(f"文件复制完成: 总计 {stats['total']}, 成功 {stats['success']}, 失败 {stats['failed']}")
        
        return stats
    
    def process_data_preparation(self) -> Dict:
        """
        执行完整的数据准备流程
        
        Returns:
            处理结果统计
        """
        logger.info("=== 开始数据准备与组织 ===")
        
        # 1. 扫描良性样本
        logger.info("扫描良性样本...")
        benign_samples = self.scan_samples(self.benign_dir, label=0)
        
        # 2. 扫描恶意样本
        logger.info("扫描恶意样本...")
        malware_samples = self.scan_samples(self.malware_dir, label=1)
        
        # 3. 合并所有样本
        all_samples = benign_samples + malware_samples
        
        if len(all_samples) == 0:
            logger.error("没有找到任何样本文件")
            return {'status': 'failed', 'message': '没有找到任何样本文件'}
        
        # 4. 分配数据集划分
        logger.info("分配数据集划分...")
        all_samples = self.assign_splits(all_samples)
        
        # 5. 创建labels.csv
        logger.info("创建labels.csv文件...")
        csv_path = self.create_labels_csv(all_samples)
        
        # 6. 复制样本到目标目录
        logger.info("复制样本到目标目录...")
        copy_stats = self.copy_samples_to_target(all_samples)
        
        # 7. 返回结果
        result = {
            'status': 'success',
            'csv_file': csv_path,
            'total_samples': len(all_samples),
            'benign_count': len(benign_samples),
            'malware_count': len(malware_samples),
            'copy_stats': copy_stats,
            'target_directory': self.target_dir
        }
        
        logger.info("=== 数据准备与组织完成 ===")
        
        return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='阶段一：数据准备与组织')
    parser.add_argument('--benign-dir',  default='D:/Test/AAAAAAAAAAAAAAA/demooooooo/dataset/benign', 
                       help='良性样本目录路径 (a文件夹)')
    parser.add_argument('--malware-dir',  default='D:/Test/AAAAAAAAAAAAAAA/demooooooo/dataset/malware', 
                       help='恶意样本目录路径 (b文件夹)')
    parser.add_argument('--target-dir', default='D:/Test/AAAAAAAAAAAAAAA/demooooooo/dataset/all_data',
                       help='目标数据目录 (指定目录)')
    parser.add_argument('--output-dir', default='d:/Test/AAAAAAAAAAAAAAA/demooooooo/output',
                       help='输出目录 (CSV文件保存位置)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子，用于数据集划分的可重现性')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 检查参数
    if not args.benign_dir and not args.malware_dir:
        print("警告: 良性样本目录和恶意样本目录都为空")
        print("请使用 -a 参数指定良性样本目录，使用 -b 参数指定恶意样本目录")
        print("示例: python data_preparation.py -a /path/to/benign -b /path/to/malware")
        return
    
    # 创建数据准备器
    data_prep = DataPreparation(
        benign_dir=args.benign_dir,
        malware_dir=args.malware_dir,
        target_dir=args.target_dir,
        output_dir=args.output_dir
    )
    
    # 执行数据准备
    result = data_prep.process_data_preparation()
    
    # 输出结果
    print(f"\n=== 数据准备结果 ===")
    print(f"状态: {result['status']}")
    
    if result['status'] == 'success':
        print(f"CSV文件: {result['csv_file']}")
        print(f"总样本数: {result['total_samples']}")
        print(f"良性样本: {result['benign_count']}")
        print(f"恶意样本: {result['malware_count']}")
        print(f"目标目录: {result['target_directory']}")
        print(f"文件复制: 成功 {result['copy_stats']['success']}, 失败 {result['copy_stats']['failed']}")
    else:
        print(f"错误: {result.get('message', '未知错误')}")

if __name__ == "__main__":
    main()