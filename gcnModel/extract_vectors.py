#!/usr/bin/env python3
"""
特征向量提取脚本
从训练好的CFG和FCG模型中提取特征向量
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from tqdm import tqdm

# 添加公共模块路径
sys.path.append(str(Path(__file__).parent))
from common_model_components import BaseGraphClassifier, create_model_from_config
from common_config import BaseConfig, CFGConfig, FCGConfig
# 导入通用数据集类
from unified_training_framework import PtLabelledDataset

class FeatureExtractor:
    """特征提取器基类"""
    
    def __init__(self, model_path: str, config, device: str = 'auto'):
        """
        初始化特征提取器
        
        Args:
            model_path: 训练好的模型路径
            config: 配置对象
            device: 设备类型 ('auto', 'cpu', 'cuda')
        """
        self.config = config
        self.device = self._setup_device(device)
        
        # 检查HDF5可用性
        self.hdf5_available = self._check_hdf5_availability()
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"特征提取器初始化完成，使用设备: {self.device}")
        if self.hdf5_available:
            print("HDF5支持可用")
        else:
            print("HDF5不可用，将使用NPZ格式")
    
    def _check_hdf5_availability(self) -> bool:
        """检查HDF5是否可用"""
        try:
            import h5py
            return True
        except Exception as e:
            print(f"HDF5导入失败: {e}")
            return False
    
    def _setup_device(self, device: str) -> torch.device:
        """设置设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def _load_model(self, model_path: str) -> BaseGraphClassifier:
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # [修改] 使用通用工厂函数创建模型
        model = create_model_from_config(self.config, model_type='classifier')
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 处理不同的检查点格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def extract_features(self, data_loader: DataLoader, 
                        output_file: str = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        提取特征向量
        
        Args:
            data_loader: 数据加载器
            output_file: 输出文件路径（可选）
            
        Returns:
            features: 特征向量数组 (N, embedding_dim)
            labels: 标签数组 (N,)
            filenames: 文件名列表
        """
        features_list = []
        labels_list = []
        filenames_list = []
        
        print("开始提取特征向量...")
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="提取特征"):
                batch = batch.to(self.device)
                
                # 获取嵌入向量
                embeddings = self.model.get_embedding(batch.x, batch.edge_index, batch.batch)
                
                # 收集结果
                features_list.append(embeddings.cpu().numpy())
                labels_list.append(batch.y.cpu().numpy())
                
                # 收集文件名（如果有的话）
                if hasattr(batch, 'filename'):
                    filenames_list.extend(batch.filename)
                else:
                    # 生成默认文件名
                    batch_size = batch.y.size(0)
                    filenames_list.extend([f"sample_{len(filenames_list) + i}" for i in range(batch_size)])
        
        # 合并结果
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        
        print(f"提取完成，共 {len(features)} 个样本，特征维度: {features.shape[1]}")
        
        # 保存结果
        if output_file:
            self._save_features(features, labels, filenames_list, output_file)
        
        return features, labels, filenames_list
    
    def _save_features(self, features: np.ndarray, labels: np.ndarray, 
                      filenames: List[str], output_file: str):
        """
        保存特征到HDF5文件
        
        Args:
            features: 特征矩阵 (N, feature_dim)
            labels: 标签数组 (N,)
            filenames: 文件名列表
            output_file: 输出文件路径
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.hdf5_available:
            raise RuntimeError("HDF5不可用，无法保存特征文件")
        
        if not output_file.endswith('.h5'):
            raise ValueError("输出文件必须是.h5格式")
        
        # HDF5格式保存
        import h5py  # 动态导入
        print(f"保存特征到HDF5文件: {output_file}")
        
        with h5py.File(output_file, 'w') as f:
            # 保存数据集
            f.create_dataset('features', data=features, compression='gzip', compression_opts=9)
            f.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)
            
            # 将文件名转换为字节串数组以便HDF5存储
            sample_id = np.array([fname.encode('utf-8') for fname in filenames])
            f.create_dataset('sample_id', data=sample_id, compression='gzip', compression_opts=9)
            
            # 添加元数据
            f.attrs['num_samples'] = len(features)
            f.attrs['feature_dim'] = features.shape[1]
            f.attrs['num_classes'] = len(np.unique(labels))
            f.attrs['description'] = 'Extracted features from GCN model'
            f.attrs['format_version'] = '1.0'
            
        print(f"HDF5文件保存完成: {features.shape[0]} 个样本, {features.shape[1]} 维特征")
            
    def test_h5_file(self, h5_path: str):
        """
        测试HDF5文件的内容
        
        Args:
            h5_path: HDF5文件路径
        """
        if not self.hdf5_available:
            print("HDF5不可用，跳过文件测试")
            return
            
        if not Path(h5_path).exists():
            print(f"文件不存在: {h5_path}")
            return
            
        try:
            import h5py  # 动态导入
            with h5py.File(h5_path, 'r') as f:
                print(f"\n=== HDF5文件内容测试: {h5_path} ===")
                print(f"数据集: {list(f.keys())}")
                
                if 'features' in f:
                    print(f"特征形状: {f['features'].shape}")
                    print(f"特征数据类型: {f['features'].dtype}")
                    
                if 'labels' in f:
                    print(f"标签形状: {f['labels'].shape}")
                    print(f"标签数据类型: {f['labels'].dtype}")
                    print(f"唯一标签: {np.unique(f['labels'][:])}")
                    
                if 'sample_id' in f:
                    print(f"样本ID形状: {f['sample_id'].shape}")
                    print(f"前3个样本ID: {[sid.decode('utf-8') for sid in f['sample_id'][:3]]}")
                
                print("元数据:")
                for key, value in f.attrs.items():
                    print(f"  {key}: {value}")
                    
        except Exception as e:
            print(f"读取HDF5文件时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='特征向量提取工具')
    parser.add_argument('--model_type', type=str, choices=['cfg', 'fcg'], 
                       required=True, help='模型类型')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='训练好的模型路径')
    # [新增] 要提取哪个数据集划分
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'],
                       default='all', help='要提取的数据集划分 (all 表示合并所有)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='设备类型')
    
    args = parser.parse_args()

    # 1. 根据模型类型选择配置和输出路径
    if args.model_type == 'cfg':
        config = CFGConfig()
        data_dirs = [config.BENIGN_CFG_DIR, config.MALWARE_CFG_DIR]
        suffix = config.CFG_SUFFIX
        output_file = f"/mnt/data1_l20_raid5disk/lbq_dataset/output/feature/cfg_extracted_features_{args.split}.h5"
    
    elif args.model_type == 'fcg':
        config = FCGConfig()
        data_dirs = [config.BENIGN_FCG_DIR, config.MALWARE_FCG_DIR]
        suffix = config.FCG_SUFFIX
        output_file = f"/mnt/data1_l20_raid5disk/lbq_dataset/output/feature/fcg_extracted_features_{args.split}.h5"

    print(f"=== {args.model_type.upper()} 特征提取开始 ===")
    print(f"输出文件: {output_file}")

    # 2. 创建特征提取器
    extractor = FeatureExtractor(
        model_path=args.model_path,
        config=config,
        device=args.device
    )

    # 3. 创建数据加载器
    if args.split == 'all':
        # 如果是 'all'，加载所有三个划分并合并
        print("加载所有数据 (train, val, test)...")
        train_dataset = PtLabelledDataset(str(config.LABELS_CSV), data_dirs, 'train', suffix)
        val_dataset = PtLabelledDataset(str(config.LABELS_CSV), data_dirs, 'val', suffix)
        test_dataset = PtLabelledDataset(str(config.LABELS_CSV), data_dirs, 'test', suffix)
        full_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset, test_dataset])
        if not full_dataset.datasets:  # 检查是否为空
            print("错误: 无法加载任何数据。请检查CSV和.pt文件。")
            return
    else:
        print(f"加载 {args.split} 数据...")
        full_dataset = PtLabelledDataset(str(config.LABELS_CSV), data_dirs, args.split, suffix)
        if len(full_dataset) == 0:
            print(f"错误: 无法加载 {args.split} 数据。请检查CSV和.pt文件。")
            return
    
    data_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 提取时建议为0
        pin_memory=True if args.device == 'cuda' else False
    )

    # 4. 提取特征
    features, labels, filenames = extractor.extract_features(
        data_loader=data_loader,
        output_file=output_file
    )
    
    # 5. 测试HDF5文件读取
    extractor.test_h5_file(output_file)
    
    print(f"\n=== {args.model_type.upper()} 特征提取完成! ===")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    main()