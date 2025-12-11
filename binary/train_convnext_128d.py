#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvNeXt 128维特征训练脚本
基于预训练的ConvNeXtV2-Tiny构建自定义投影模型，训练生成128维特征向量
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import numpy as np
import pandas as pd
import os
import glob
import json
import subprocess
import time
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# === GPU 自动检测和管理工具 ===

def get_gpu_memory_usage():
    """获取所有GPU的显存使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                gpu_id = int(parts[0])
                memory_used = int(parts[1])
                memory_total = int(parts[2])
                gpu_util = int(parts[3])
                memory_usage_percent = (memory_used / memory_total) * 100
                
                gpu_info.append({
                    'gpu_id': gpu_id,
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'memory_free': memory_total - memory_used,
                    'memory_usage_percent': memory_usage_percent,
                    'gpu_utilization': gpu_util
                })
        
        return gpu_info
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"警告: 无法获取GPU信息: {e}")
        return []

def find_best_gpu(min_free_memory_gb=2, max_memory_usage_percent=80):
    """
    找到最适合的GPU
    
    Args:
        min_free_memory_gb: 最小空闲显存要求(GB)
        max_memory_usage_percent: 最大显存使用率阈值
    
    Returns:
        最佳GPU的ID，如果没有合适的GPU则返回None
    """
    gpu_info = get_gpu_memory_usage()
    
    if not gpu_info:
        print("未检测到GPU信息，使用默认设备")
        return None
    
    # 打印所有GPU状态
    print("\n=== GPU 状态信息 ===")
    for gpu in gpu_info:
        print(f"GPU {gpu['gpu_id']}: "
              f"显存 {gpu['memory_used']}/{gpu['memory_total']} MB "
              f"({gpu['memory_usage_percent']:.1f}%), "
              f"利用率 {gpu['gpu_utilization']}%, "
              f"空闲显存 {gpu['memory_free']} MB")
    
    # 筛选符合条件的GPU
    suitable_gpus = []
    min_free_memory_mb = min_free_memory_gb * 1024
    
    for gpu in gpu_info:
        if (gpu['memory_free'] >= min_free_memory_mb and 
            gpu['memory_usage_percent'] <= max_memory_usage_percent):
            suitable_gpus.append(gpu)
    
    if not suitable_gpus:
        print(f"警告: 没有找到符合条件的GPU (需要至少{min_free_memory_gb}GB空闲显存，使用率不超过{max_memory_usage_percent}%)")
        # 如果没有完全符合条件的，选择空闲显存最多的
        best_gpu = max(gpu_info, key=lambda x: x['memory_free'])
        print(f"选择空闲显存最多的GPU: {best_gpu['gpu_id']}")
        return best_gpu['gpu_id']
    
    # 选择空闲显存最多的符合条件的GPU
    best_gpu = max(suitable_gpus, key=lambda x: x['memory_free'])
    print(f"选择最佳GPU: {best_gpu['gpu_id']} (空闲显存: {best_gpu['memory_free']} MB)")
    
    return best_gpu['gpu_id']

def setup_device(gpu_id=None):
    """
    设置训练设备
    
    Args:
        gpu_id: 指定的GPU ID，如果为None则自动选择
    
    Returns:
        torch.device对象
    """
    if not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        return torch.device("cpu")
    
    if gpu_id is None:
        gpu_id = find_best_gpu()
    
    if gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        print(f"使用GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        
        # 清理GPU缓存
        torch.cuda.empty_cache()
        
        return device
    else:
        device = torch.device("cuda")
        print(f"使用默认GPU: {torch.cuda.get_device_name()}")
        return device

def monitor_gpu_memory(device):
    """监控GPU显存使用情况"""
    if device.type == 'cuda':
        gpu_id = device.index if device.index is not None else 0
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3   # GB
        print(f"GPU {gpu_id} 显存使用: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")

def get_optimal_num_workers():
    """
    智能确定最优的DataLoader worker数量
    采用极度保守的策略以避免共享内存问题
    
    Returns:
        int: 建议的worker数量
    """
    try:
        import psutil
        
        # 获取系统信息
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().available / (1024**3)
        
        print(f"系统信息: CPU核心数={cpu_count}, 可用内存={memory_gb:.1f}GB")
        
        # 极度保守策略：在Linux服务器环境中，共享内存限制通常很严格
        # 即使有大量内存，/dev/shm 的大小也可能有限
        
        # 检查是否在Linux环境中
        import platform
        is_linux = platform.system().lower() == 'linux'
        
        if is_linux:
            # Linux环境：由于/dev/shm限制，使用更保守的策略
            print("检测到Linux环境，使用保守的worker配置")
            if memory_gb < 16:
                return 0  # 内存不足16GB，强制单进程
            elif memory_gb < 64:
                return 1  # 内存64GB以下，最多1个worker
            elif memory_gb < 128:
                return 2  # 内存128GB以下，最多2个worker
            else:
                return min(2, max(1, cpu_count // 24))  # 大内存服务器，仍然保守
        else:
            # Windows/Mac环境：相对宽松的策略
            if memory_gb < 8:
                return 0
            elif memory_gb < 16:
                return min(2, max(1, cpu_count // 8))
            else:
                return min(4, max(2, cpu_count // 6))
            
    except ImportError:
        print("警告: psutil未安装，使用单进程模式")
        return 0
    except Exception as e:
        print(f"警告: 获取系统信息失败 ({e})，使用单进程模式")
        return 0

def safe_dataloader_creation(dataset, batch_size, shuffle=False, optimal_workers=0, collate_fn=None):
    """
    安全创建DataLoader，自动处理共享内存问题
    采用多层级降级策略确保成功创建
    
    Args:
        dataset: 数据集
        batch_size: 批大小
        shuffle: 是否打乱
        optimal_workers: 建议的worker数量
        collate_fn: 自定义collate函数
    
    Returns:
        DataLoader对象
    """
    import platform
    is_linux = platform.system().lower() == 'linux'
    
    # 在Linux环境中，由于共享内存限制更严格，采用更保守的策略
    if is_linux and optimal_workers > 0:
        print(f"Linux环境检测到，将worker数量从 {optimal_workers} 降级为单进程模式")
        optimal_workers = 0
    
    # 尝试不同的配置直到成功
    configs_to_try = [
        # 配置1: 单进程模式（最安全）
        {
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': None
        }
    ]
    
    # 只有在非Linux环境且optimal_workers > 0时才尝试多进程
    if not is_linux and optimal_workers > 0:
        configs_to_try.insert(0, {
            'num_workers': 1,  # 最多1个worker
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': 2
        })
    
    for i, config in enumerate(configs_to_try):
        try:
            print(f"尝试DataLoader配置 {i+1}: workers={config['num_workers']}")
            
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn,
                **config
            )
            
            # 测试加载多个batch以确保稳定性
            test_iter = iter(loader)
            for test_batch in range(min(3, len(loader))):
                try:
                    batch_data = next(test_iter)
                    # 确保数据可以正常访问
                    if hasattr(batch_data, '__len__') and len(batch_data) > 0:
                        pass  # 数据正常
                except StopIteration:
                    break
                except Exception as e:
                    raise RuntimeError(f"数据加载测试失败: {e}")
            
            print(f"DataLoader配置 {i+1} 创建成功")
            return loader
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"DataLoader配置 {i+1} 失败: {e}")
            
            # 检查是否是共享内存相关错误
            if any(keyword in error_msg for keyword in ['shared memory', 'shm', 'bus error', 'worker']):
                print("检测到共享内存相关错误，强制使用单进程模式")
                # 如果还没有尝试单进程模式，直接跳到单进程配置
                if config['num_workers'] > 0:
                    continue
            
            if i == len(configs_to_try) - 1:
                raise RuntimeError(f"所有DataLoader配置都失败了，最后的错误: {e}")
            continue

def test_dataloader_stability(loader, name="DataLoader", max_batches=5):
    """
    测试DataLoader的稳定性，确保在训练过程中不会出现共享内存错误
    
    Args:
        loader: DataLoader对象
        name: DataLoader名称（用于日志）
        max_batches: 最大测试批次数
    
    Returns:
        bool: 测试是否通过
    """
    print(f"\n=== 测试 {name} 稳定性 ===")
    try:
        test_count = 0
        for batch_idx, batch_data in enumerate(loader):
            if batch_idx >= max_batches:
                break
                
            # 测试数据访问
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                images, labels = batch_data[0], batch_data[1]
                
                # 确保数据可以正常访问
                _ = images.shape
                _ = labels.shape
                
                # 测试数据移动到GPU（如果需要）
                if torch.cuda.is_available():
                    try:
                        images_gpu = images.cuda(non_blocking=True)
                        labels_gpu = labels.cuda(non_blocking=True)
                        # 立即释放GPU内存
                        del images_gpu, labels_gpu
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"GPU数据移动测试失败: {e}")
                        
                test_count += 1
                
            if batch_idx % 2 == 0:
                print(f"  批次 {batch_idx + 1}/{min(max_batches, len(loader))} 测试通过")
        
        print(f"{name} 稳定性测试完成，成功测试 {test_count} 个批次")
        return True
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"{name} 稳定性测试失败: {e}")
        
        # 检查是否是共享内存相关错误
        if any(keyword in error_msg for keyword in ['shared memory', 'shm', 'bus error', 'worker']):
            print(f"检测到共享内存错误，{name} 不稳定")
            return False
        else:
            print(f"其他错误: {e}")
            return False

def cleanup_gpu_memory(device):
    """清理GPU显存"""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print("已清理GPU缓存")

# === 1. 配置 ===

# 路径配置 (Ubuntu路径)
# CSV_PATH = "/mnt/data1_l20_raid5disk/lbq_dataset/output/labels.csv"
# BENIGN_NPY_DIR = "/mnt/data1_l20_raid5disk/lbq_dataset/dataset/benign_RGB"
# MALWARE_NPY_DIR = "/mnt/data1_l20_raid5disk/lbq_dataset/dataset/malware_RGB"
# PRETRAINED_PATH = "/mnt/data1_l20_raid5disk/lbq_dataset/model/convnextv2_tiny_22k_224_ema.pt"
# OUTPUT_DIR = "/mnt/data1_l20_raid5disk/lbq_dataset/output/convnext"
# BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "convnext_128d_best_model.pth")
# 路径配置 (Ubuntu路径)
CSV_PATH = "/mnt/lbq/output/labels.csv"
BENIGN_NPY_DIR = "/mnt/lbq/dataset/benign_RGB"
MALWARE_NPY_DIR = "/mnt/lbq/dataset/malware_RGB"
PRETRAINED_PATH = "/mnt/lbq/model/convnextv2_tiny_22k_224_ema.pt"
OUTPUT_DIR = "/mnt/lbq/output/convnext"
BEST_MODEL_PATH = "/mnt/lbq/output/convnext/convnext_128d_best_model.pth"


# 模型配置
MODEL_NAME = 'convnextv2_tiny.fcmae_ft_in22k_in1k' 
FEATURE_DIM_IN = 768  # ConvNeXt-Tiny 输出维度
FEATURE_DIM = 128 # 目标特征维度
NUM_CLASSES = 2

# 训练参数
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # 修改为0以避免共享内存问题，使用主进程加载数据
# DEVICE 将在 main() 函数中通过自动检测设置

# === 2. 自定义模型架构 ===

class ConvNextWithProjection(nn.Module):
    """
    ConvNeXt + 128维投影头 + 分类头的自定义模型
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

# === 3. 数据集类 ===

class NpyPeDatasetFromCsv(Dataset):
    """
    从CSV文件和NPY文件夹加载PE特征图数据集
    """
    def __init__(self, df, benign_dir, malware_dir, transform=None):
        self.file_paths = []
        self.labels = []
        self.transform = transform
        
        print(f"正在构建数据集，共 {len(df)} 个样本...")
        
        valid_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="检查文件"):
            label = int(row['label'])
            sample_id = row['sample_id']
            
            # 根据标签生成不同格式的文件名
            if label == 0:  # 良性软件
                safe_filename = sample_id.replace(os.sep, '_') + '.exe.npy'
                full_path = os.path.join(benign_dir, safe_filename)
            else:  # 恶意软件
                safe_filename = sample_id.replace(os.sep, '_') + '.npy'
                full_path = os.path.join(malware_dir, safe_filename)
                
            if os.path.exists(full_path):
                self.file_paths.append(full_path)
                self.labels.append(label)
                valid_count += 1
        
        print(f"数据集构建完成: {valid_count}/{len(df)} 个有效样本")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # 加载NPY文件
            image = np.load(file_path)
            
            # 转换为PyTorch张量并归一化到[0,1]
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            # 返回无效标记
            return torch.zeros((3, 224, 224)), torch.tensor(-1, dtype=torch.long)

def collate_fn(batch):
    """自定义批处理函数，过滤无效样本"""
    batch = list(filter(lambda x: x[1] != -1, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

# === 4. 训练和评估函数 ===

def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels in loop:
        if images.shape[0] == 0: 
            continue
            
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        loop.set_postfix(loss=loss.item())
    
    return total_loss / max(num_batches, 1)

def evaluate(model, loader, criterion, device, desc="Evaluating"):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    num_batches = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            if images.shape[0] == 0: 
                continue
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    avg_loss = total_loss / max(num_batches, 1)
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0) if all_labels else 0.0
    precision = precision_score(all_labels, all_preds, zero_division=0) if all_labels else 0.0
    recall = recall_score(all_labels, all_preds, zero_division=0) if all_labels else 0.0
    cm = confusion_matrix(all_labels, all_preds) if all_labels else np.array([[0, 0], [0, 0]])
    
    return avg_loss, acc, f1, precision, recall, cm

# === 5. 主训练逻辑 ===

def main():
    """主训练函数"""
    print(f"=== ConvNeXt 128维特征训练开始 ===")
    
    # 自动选择最佳GPU
    DEVICE = setup_device()
    
    # 智能配置worker数量
    try:
        optimal_workers = get_optimal_num_workers()
    except ImportError:
        print("警告: psutil未安装，使用默认配置 NUM_WORKERS=0")
        optimal_workers = 0
    except Exception as e:
        print(f"警告: 获取系统信息失败 {e}，使用默认配置 NUM_WORKERS=0")
        optimal_workers = 0
    
    print(f"批大小: {BATCH_SIZE}, 训练轮数: {EPOCHS}, Worker数量: {optimal_workers}")
    
    # 监控初始GPU状态
    monitor_gpu_memory(DEVICE)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 数据预处理
    data_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集划分
    print(f"正在加载数据集: {CSV_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"标签文件不存在: {CSV_PATH}")
        
    df = pd.read_csv(CSV_PATH)
    print(f"总样本数: {len(df)}")
    
    # 按split列划分数据集
    df_train = df[df['split'] == 'train'].reset_index(drop=True)
    df_val = df[df['split'] == 'val'].reset_index(drop=True)
    df_test = df[df['split'] == 'test'].reset_index(drop=True)
    
    print(f"训练集: {len(df_train)}, 验证集: {len(df_val)}, 测试集: {len(df_test)}")
    
    # 创建数据集
    train_dataset = NpyPeDatasetFromCsv(df_train, BENIGN_NPY_DIR, MALWARE_NPY_DIR, transform=data_transform)
    val_dataset = NpyPeDatasetFromCsv(df_val, BENIGN_NPY_DIR, MALWARE_NPY_DIR, transform=data_transform)
    test_dataset = NpyPeDatasetFromCsv(df_test, BENIGN_NPY_DIR, MALWARE_NPY_DIR, transform=data_transform)
    
    # 创建数据加载器 - 使用安全的创建方法
    print("\n=== 创建训练数据加载器 ===")
    train_loader = safe_dataloader_creation(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        optimal_workers=optimal_workers,
        collate_fn=collate_fn
    )
    
    print("\n=== 创建验证数据加载器 ===")
    val_loader = safe_dataloader_creation(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        optimal_workers=optimal_workers,
        collate_fn=collate_fn
    )
    
    print("\n=== 创建测试数据加载器 ===")
    test_loader = safe_dataloader_creation(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        optimal_workers=optimal_workers,
        collate_fn=collate_fn
    )
    
    # 进行DataLoader稳定性测试
    print("\n=== DataLoader 稳定性测试 ===")
    train_stable = test_dataloader_stability(train_loader, "训练DataLoader", max_batches=3)
    val_stable = test_dataloader_stability(val_loader, "验证DataLoader", max_batches=2)
    test_stable = test_dataloader_stability(test_loader, "测试DataLoader", max_batches=2)
    
    if not all([train_stable, val_stable, test_stable]):
        print("\n警告: DataLoader稳定性测试失败，强制重新创建为单进程模式")
        
        # 强制使用单进程模式重新创建
        print("重新创建训练DataLoader（单进程模式）")
        train_loader = safe_dataloader_creation(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            optimal_workers=0,  # 强制单进程
            collate_fn=collate_fn
        )
        
        print("重新创建验证DataLoader（单进程模式）")
        val_loader = safe_dataloader_creation(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            optimal_workers=0,  # 强制单进程
            collate_fn=collate_fn
        )
        
        print("重新创建测试DataLoader（单进程模式）")
        test_loader = safe_dataloader_creation(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            optimal_workers=0,  # 强制单进程
            collate_fn=collate_fn
        )
        
        # 再次测试稳定性
        print("\n=== 单进程模式稳定性验证 ===")
        train_stable = test_dataloader_stability(train_loader, "训练DataLoader（单进程）", max_batches=2)
        if not train_stable:
            raise RuntimeError("即使在单进程模式下，DataLoader仍然不稳定，请检查数据集或系统配置")
    
    print("\n所有DataLoader创建并验证完成！")
    
    # 构建模型
    print("正在构建ConvNeXt + 128维投影模型...")
        
    model = ConvNextWithProjection(MODEL_NAME, num_classes=NUM_CLASSES)
    
    # 如果有自定义预训练权重，可以在这里加载
    if os.path.exists(PRETRAINED_PATH):
        print(f"加载自定义预训练权重: {PRETRAINED_PATH}")
        try:
            checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')
            # 尝试不同的键名
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 只加载backbone部分的权重
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.') or not key.startswith(('projection_head.', 'classifier_head.')):
                    # 移除可能的前缀
                    new_key = key.replace('backbone.', '') if key.startswith('backbone.') else key
                    backbone_state_dict[new_key] = value
            
            # 加载权重，允许部分匹配
            missing_keys, unexpected_keys = model.backbone.load_state_dict(backbone_state_dict, strict=False)
            if missing_keys:
                print(f"警告: 缺少的键: {len(missing_keys)} 个")
            if unexpected_keys:
                print(f"警告: 意外的键: {len(unexpected_keys)} 个")
            print("自定义预训练权重加载完成")
        except Exception as e:
            print(f"加载自定义预训练权重失败: {e}")
            print("将使用timm默认预训练权重")
    else:
        print("使用timm默认预训练权重")
    
    model.to(DEVICE)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 训练配置信息
    training_config = {
        'model_name': MODEL_NAME,
        'feature_dim_in': FEATURE_DIM_IN,
        'feature_dim': FEATURE_DIM,
        'num_classes': NUM_CLASSES,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'num_workers': NUM_WORKERS,
        'device': str(DEVICE),
        'csv_path': CSV_PATH,
        'benign_npy_dir': BENIGN_NPY_DIR,
        'malware_npy_dir': MALWARE_NPY_DIR,
        'pretrained_path': PRETRAINED_PATH,
        'output_dir': OUTPUT_DIR,
        'training_start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'train_samples': len(df_train),
            'val_samples': len(df_val),
            'test_samples': len(df_test)
        }
    }
    
    # 训练历史记录
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'val_accuracies': [],
        'val_f1_scores': [],
        'val_precisions': [],
        'val_recalls': [],
        'learning_rates': [],
        'epochs': []
    }
    
    # 训练循环
    best_f1 = 0.0
    best_epoch = 0
    print("\n=== 开始训练 ===")
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # 验证
        val_loss, acc, f1, precision, recall, cm = evaluate(model, val_loader, criterion, DEVICE, desc="验证中")
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 记录训练历史
        training_history['train_losses'].append(train_loss)
        training_history['val_losses'].append(val_loss)
        training_history['val_accuracies'].append(acc)
        training_history['val_f1_scores'].append(f1)
        training_history['val_precisions'].append(precision)
        training_history['val_recalls'].append(recall)
        training_history['learning_rates'].append(current_lr)
        training_history['epochs'].append(epoch + 1)
        
        print(f"训练损失: {train_loss:.4f}")
        print(f"验证损失: {val_loss:.4f}")
        print(f"验证指标: F1={f1:.4f}, Acc={acc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")
        print(f"学习率: {current_lr:.6f}")
        
        # 监控GPU显存使用情况
        monitor_gpu_memory(DEVICE)
        
        # 保存最佳模型和完整信息
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            print(f"*** 新的最佳F1分数: {f1:.4f} ! 保存模型到 {BEST_MODEL_PATH} ***")
            
            # 清理GPU缓存以释放显存
            cleanup_gpu_memory(DEVICE)
            
            # 保存完整的模型检查点
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'val_loss': val_loss,
                'val_accuracy': acc,
                'val_precision': precision,
                'val_recall': recall,
                'confusion_matrix': cm.tolist(),
                'training_config': training_config,
                'training_history': training_history.copy()
            }
            torch.save(checkpoint, BEST_MODEL_PATH)
            
            # 保存训练配置为单独的JSON文件
            config_path = os.path.join(OUTPUT_DIR, "training_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(training_config, f, indent=2, ensure_ascii=False)
    
    # 保存完整的训练历史
    training_config['training_end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    training_config['best_epoch'] = best_epoch
    training_config['best_f1_score'] = best_f1
    
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    final_history = {
        'config': training_config,
        'history': training_history
    }
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(final_history, f, indent=2, ensure_ascii=False)
    
    # 最终测试
    print("\n=== 训练完成，进行最终测试 ===")
    print(f"加载最佳模型: {BEST_MODEL_PATH}")
    
    # 清理GPU缓存
    cleanup_gpu_memory(DEVICE)
    
    # 加载完整的检查点
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"最佳模型来自第 {checkpoint['epoch']} 轮，F1分数: {checkpoint['best_f1']:.4f}")
    else:
        # 兼容旧格式（只有state_dict）
        model.load_state_dict(checkpoint)
    
    # 监控测试前的GPU状态
    monitor_gpu_memory(DEVICE)
    
    test_loss, acc, f1, precision, recall, cm = evaluate(model, test_loader, criterion, DEVICE, desc="最终测试")
    
    # 保存最终测试结果
    final_results = {
        'test_loss': test_loss,
        'test_accuracy': acc,
        'test_f1_score': f1,
        'test_precision': precision,
        'test_recall': recall,
        'confusion_matrix': cm.tolist(),
        'test_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = os.path.join(OUTPUT_DIR, "final_test_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    print("\n=== 最终测试结果 ===")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试F1分数: {f1:.4f}")
    print(f"测试准确率: {acc:.4f}")
    print(f"测试精确率: {precision:.4f}")
    print(f"测试召回率: {recall:.4f}")
    print(f"混淆矩阵:\n{cm}")
    
    print(f"\n=== 训练完成! ===")
    print(f"最佳模型已保存至: {BEST_MODEL_PATH}")
    print(f"训练配置已保存至: {os.path.join(OUTPUT_DIR, 'training_config.json')}")
    print(f"训练历史已保存至: {os.path.join(OUTPUT_DIR, 'training_history.json')}")
    print(f"测试结果已保存至: {os.path.join(OUTPUT_DIR, 'final_test_results.json')}")
    
    # 最终清理GPU显存
    print(f"\n=== 清理GPU资源 ===")
    cleanup_gpu_memory(DEVICE)
    monitor_gpu_memory(DEVICE)

if __name__ == "__main__":
    main()