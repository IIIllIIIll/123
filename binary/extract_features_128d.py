#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvNeXt 128维特征提取脚本
从训练好的ConvNeXt模型中提取128维特征向量，并保存为HDF5格式
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import numpy as np
import pandas as pd
import h5py
import os
from tqdm import tqdm

# === 1. 配置 ===

# 路径配置 (Ubuntu路径)
# CSV_PATH = "/mnt/data1_l20_raid5disk/lbq_dataset/output/labels.csv"
# BENIGN_NPY_DIR = "/mnt/data1_l20_raid5disk/lbq_dataset/dataset/benign_RGB"
# MALWARE_NPY_DIR = "/mnt/data1_l20_raid5disk/lbq_dataset/dataset/malware_RGB"
# PRETRAINED_PATH = "/mnt/data1_l20_raid5disk/lbq_dataset/model/convnextv2_tiny_22k_224_ema.pt"
# TRAINED_MODEL_PATH = "/mnt/data1_l20_raid5disk/lbq_dataset/output/convnext/convnext_128d_best_model.pth"
# OUTPUT_H5_PATH = "/mnt/data1_l20_raid5disk/lbq_dataset/output/feature/convnext_extracted_features_128d.h5"

# 路径配置 (Ubuntu路径)
CSV_PATH = "/mnt/lbq/output/labels.csv"
BENIGN_NPY_DIR = "/mnt/lbq/dataset/benign_RGB"
MALWARE_NPY_DIR = "/mnt/lbq/dataset/malware_RGB"
PRETRAINED_PATH = "/mnt/lbq/model/convnextv2_tiny_22k_224_ema.pt"
# TRAINED_MODEL_PATH = "/mnt/lbq/output/convnext/convnext_128d_best_model.pth"
TRAINED_MODEL_PATH = "/mnt/lbq/output/convnext-16/convnext_128d_best_model.pth"
OUTPUT_H5_PATH = "/mnt/lbq/output/feature/convnext_extracted_features_128d.h5"

# 模型配置
MODEL_NAME = 'convnextv2_tiny.fcmae_ft_in22k_in1k' 
FEATURE_DIM_IN = 768  # ConvNeXt-Tiny 输出维度
FEATURE_DIM = 128 # 目标特征维度
NUM_CLASSES = 2

# 处理参数
BATCH_SIZE = 16
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# === 2. 模型架构 (与训练脚本保持一致) ===

class ConvNextWithProjection(nn.Module):
    """
    ConvNeXt模型 + 投影头 + 分类头
    """
    def __init__(self, model_name, num_classes=2):
        super().__init__()
        
        # 1. 加载预训练主干网络 (无分类头)
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True,  # 使用timm默认预训练权重
            num_classes=0  # 输出 (batch, 768)
        )
        
        # 2. 定义投影头和分类头
        self.projection_head = nn.Linear(FEATURE_DIM_IN, FEATURE_DIM)
        self.classifier_head = nn.Linear(FEATURE_DIM, num_classes)
        
        # 3. 初始化新添加的层
        nn.init.xavier_uniform_(self.projection_head.weight)
        nn.init.zeros_(self.projection_head.bias)
        nn.init.xavier_uniform_(self.classifier_head.weight)
        nn.init.zeros_(self.classifier_head.bias)
        
    def forward(self, x):
        """前向传播"""
        # 1. 主干网络特征提取 (batch, 768)
        x = self.backbone(x)
        
        # 2. 投影到128维 (batch, 128)
        x = self.projection_head(x) 
        
        # 3. 分类输出 (batch, 2)
        output = self.classifier_head(x)
        return output

    def extract_features(self, x):
        """特征提取函数，返回128维特征向量"""
        with torch.no_grad():
            x = self.backbone(x)
            x = self.projection_head(x)
            return x

# === 3. 特征提取函数 ===

def extract_single_feature(model, image_path, transform, device):
    """
    从单个NPY文件提取128维特征
    
    Args:
        model: 训练好的模型
        image_path: NPY文件路径
        transform: 数据预处理变换
        device: 计算设备
    
    Returns:
        numpy.ndarray: 128维特征向量，如果失败则返回零向量
    """
    try:
        # 加载NPY文件
        image = np.load(image_path)
        
        # 转换为PyTorch张量
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # 应用预处理
        if transform:
            image = transform(image)
        
        # 添加批次维度并移到设备
        image = image.unsqueeze(0).to(device)
        
        # 提取特征
        features = model.extract_features(image)
        
        # 转换为numpy数组
        return features.cpu().numpy().flatten()
        
    except Exception as e:
        print(f"提取特征失败 {image_path}: {e}")
        return np.zeros(FEATURE_DIM, dtype=np.float32)

def extract_all_features(model, df, benign_dir, malware_dir, transform, device):
    """
    提取所有样本的128维特征
    
    Args:
        model: 训练好的模型
        df: 包含样本信息的DataFrame
        benign_dir: 良性样本NPY目录
        malware_dir: 恶意样本NPY目录
        transform: 数据预处理变换
        device: 计算设备
    
    Returns:
        tuple: (features, labels, sample_ids)
    """
    model.eval()
    
    features_list = []
    labels_list = []
    sample_ids_list = []
    
    print(f"开始提取 {len(df)} 个样本的特征...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="提取特征"):
        sample_id = row['sample_id']
        label = int(row['label'])
        
        # 根据标签生成不同格式的文件名（与训练脚本保持一致）
        if label == 0:  # 良性软件
            safe_filename = sample_id.replace(os.sep, '_') + '.exe.npy'
            npy_path = os.path.join(benign_dir, safe_filename)
        else:  # 恶意软件
            safe_filename = sample_id.replace(os.sep, '_') + '.npy'
            npy_path = os.path.join(malware_dir, safe_filename)
        
        # 提取特征
        if os.path.exists(npy_path):
            feature = extract_single_feature(model, npy_path, transform, device)
        else:
            # 文件不存在，使用零向量
            feature = np.zeros(FEATURE_DIM, dtype=np.float32)
            print(f"文件不存在，使用零向量: {npy_path}")
        
        features_list.append(feature)
        labels_list.append(label)
        sample_ids_list.append(sample_id)
    
    # 转换为numpy数组
    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    sample_ids = np.array(sample_ids_list, dtype='S256')  # 字符串数组
    
    return features, labels, sample_ids

def save_features_to_h5(features, labels, sample_ids, output_path):
    """
    将特征保存到HDF5文件
    
    Args:
        features: (N, 128) 特征矩阵
        labels: (N,) 标签数组
        sample_ids: (N,) 样本ID数组
        output_path: 输出HDF5文件路径
    """
    print(f"保存特征到HDF5文件: {output_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # 保存特征矩阵
        f.create_dataset('features', data=features, compression='gzip', compression_opts=9)
        
        # 保存标签
        f.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)
        
        # 保存样本ID
        f.create_dataset('sample_id', data=sample_ids, compression='gzip', compression_opts=9)
        
        # 保存元数据
        f.attrs['num_samples'] = len(features)
        f.attrs['feature_dim'] = features.shape[1]
        f.attrs['num_classes'] = len(np.unique(labels))
        f.attrs['description'] = 'ConvNeXt 128-dimensional features extracted from PE RGB images'
    
    print(f"特征保存完成!")
    print(f"样本数量: {len(features)}")
    print(f"特征维度: {features.shape[1]}")
    print(f"文件大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

# === 4. 主函数 ===

def main():
    """主特征提取函数"""
    print("=== ConvNeXt 128维特征提取开始 ===")
    print(f"使用设备: {DEVICE}")
    
    # 检查必要文件
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"标签文件不存在: {CSV_PATH}")
    
    if not os.path.exists(TRAINED_MODEL_PATH):
        raise FileNotFoundError(f"训练好的模型不存在: {TRAINED_MODEL_PATH}")
    
    # 加载数据集信息
    print(f"加载数据集: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"总样本数: {len(df)}")
    
    # 统计标签分布
    label_counts = df['label'].value_counts().sort_index()
    print(f"标签分布: {dict(label_counts)}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 构建模型
    print("构建模型架构...")
    model = ConvNextWithProjection(MODEL_NAME, num_classes=NUM_CLASSES)
    
    # 加载训练好的权重
    print(f"加载训练好的模型权重: {TRAINED_MODEL_PATH}")
    checkpoint = torch.load(TRAINED_MODEL_PATH, map_location=DEVICE)
    
    # 检查是否是完整的checkpoint格式
    if 'model_state_dict' in checkpoint:
        print("检测到checkpoint格式，提取model_state_dict")
        state_dict = checkpoint['model_state_dict']
        # 打印一些训练信息
        if 'epoch' in checkpoint:
            print(f"模型训练轮数: {checkpoint['epoch']}")
        if 'best_f1' in checkpoint:
            print(f"最佳F1分数: {checkpoint['best_f1']:.4f}")
    else:
        print("检测到直接的state_dict格式")
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    print("模型加载完成!")
    
    # 提取所有特征
    features, labels, sample_ids = extract_all_features(
        model, df, BENIGN_NPY_DIR, MALWARE_NPY_DIR, transform, DEVICE
    )
    
    # 验证特征质量
    print("\n=== 特征质量检查 ===")
    print(f"特征矩阵形状: {features.shape}")
    print(f"特征均值: {np.mean(features):.6f}")
    print(f"特征标准差: {np.std(features):.6f}")
    print(f"零向量数量: {np.sum(np.all(features == 0, axis=1))}")
    
    # 保存到HDF5文件
    save_features_to_h5(features, labels, sample_ids, OUTPUT_H5_PATH)
    
    print("\n=== 特征提取完成! ===")
    print(f"输出文件: {OUTPUT_H5_PATH}")

def test_h5_file():
    """测试HDF5文件的读取"""
    print("\n=== 测试HDF5文件读取 ===")
    
    if not os.path.exists(OUTPUT_H5_PATH):
        print("HDF5文件不存在，跳过测试")
        return
    
    with h5py.File(OUTPUT_H5_PATH, 'r') as f:
        print("HDF5文件内容:")
        print(f"  数据集: {list(f.keys())}")
        print(f"  特征形状: {f['features'].shape}")
        print(f"  标签形状: {f['labels'].shape}")
        print(f"  样本ID形状: {f['sample_id'].shape}")
        
        print("元数据:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        # 读取前几个样本测试
        features_sample = f['features'][:5]
        labels_sample = f['labels'][:5]
        print(f"前5个样本特征形状: {features_sample.shape}")
        print(f"前5个样本标签: {labels_sample}")

if __name__ == "__main__":
    main()
    test_h5_file()