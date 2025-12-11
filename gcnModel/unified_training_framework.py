#!/usr/bin/env python3
"""
统一训练和评估框架
支持CFG和FCG两种模型的训练、验证和测试
提供通用的训练流程、评估指标和模型管理功能
"""

# 标准库导入
import argparse
import gc
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 第三方库导入
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# PyTorch相关导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch_geometric.data import DataLoader, Dataset

# Scikit-learn导入
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support, 
    roc_auc_score
)

# 本地模块导入
from common_config import BaseConfig, CFGConfig, FCGConfig, GPUManager, MemoryManager, get_adaptive_batch_config
from common_model_components import BaseDataLoader, create_model_from_config

# === 统一系统管理器 ===
class SystemManager:
    """统一系统管理器 - DataLoader管理功能"""
    
    # === DataLoader 管理功能 ===
    @staticmethod
    def get_optimal_num_workers(config_workers=None):
        """智能确定最优的DataLoader worker数量 - Linux专用
        
        Args:
            config_workers: 配置文件中指定的worker数量，如果提供则作为参考
        """
        try:
            import psutil
            
            # 获取系统信息
            cpu_count = psutil.cpu_count(logical=True)
            memory_gb = psutil.virtual_memory().available / (1024**3)
            
            print(f"系统信息: CPU核心数={cpu_count}, 可用内存={memory_gb:.1f}GB")
            
            # 如果配置文件中指定了worker数量，优先使用并验证其合理性
            if config_workers is not None:
                print(f"配置文件指定worker数量: {config_workers}")
                # 验证配置的合理性
                if config_workers == 0:
                    print("配置指定单进程模式")
                    return 0
                elif config_workers > 0:
                    # 检查系统是否支持指定的worker数量
                    if memory_gb < 8 and config_workers > 1:
                        print(f"警告: 系统内存不足({memory_gb:.1f}GB)，建议使用单进程模式")
                        print(f"但仍使用配置指定的worker数量: {config_workers}")
                    return config_workers
            
            # Linux环境：由于/dev/shm限制，使用保守的策略
            print("Linux环境，使用保守的worker配置")
            if memory_gb < 16:
                return 0  # 内存不足16GB，强制单进程
            elif memory_gb < 64:
                return 1  # 内存64GB以下，最多1个worker
            elif memory_gb < 128:
                return 2  # 内存128GB以下，最多2个worker
            else:
                return min(2, max(1, cpu_count // 24))  # 大内存服务器，仍然保守
                    
        except ImportError:
            print("警告: psutil未安装，使用单进程模式")
            return config_workers if config_workers is not None else 0
        except Exception as e:
            print(f"警告: 获取系统信息失败 ({e})，使用单进程模式")
            return config_workers if config_workers is not None else 0
    
    @staticmethod
    def create_safe_dataloader(dataset, batch_size, shuffle=False, optimal_workers=0, collate_fn=None):
        """安全创建DataLoader，自动处理共享内存问题 - Linux专用"""
        
        # 构建多种配置策略，从最优到最保守
        configs_to_try = []
        
        # 如果指定了optimal_workers，尝试多进程配置
        if optimal_workers > 0:
            # 配置1: 使用指定的worker数量（最优性能）
            configs_to_try.append({
                'num_workers': optimal_workers,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 2
            })
            
            # 配置2: 减少worker数量（中等性能）
            if optimal_workers > 2:
                configs_to_try.append({
                    'num_workers': max(1, optimal_workers // 2),
                    'pin_memory': True,
                    'persistent_workers': True,
                    'prefetch_factor': 2
                })
            
            # 配置3: 单worker模式（保守策略）
            configs_to_try.append({
                'num_workers': 1,
                'pin_memory': False,
                'persistent_workers': False,
                'prefetch_factor': 2
            })
        
        # 配置4: 单进程模式（最安全的后备方案）
        configs_to_try.append({
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': None
        })
        
        for i, config in enumerate(configs_to_try):
            try:
                # 构建配置描述
                config_desc = f"workers={config['num_workers']}"
                if config['pin_memory']:
                    config_desc += ", pin_memory=True"
                if config.get('persistent_workers'):
                    config_desc += ", persistent_workers=True"
                
                print(f"尝试DataLoader配置 {i+1}: {config_desc}")
                
                # 过滤掉None值的参数
                filtered_config = {k: v for k, v in config.items() if v is not None}
                
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    collate_fn=collate_fn,
                    **filtered_config
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
                
                print(f"DataLoader配置 {i+1} 创建成功 ({config_desc})")
                return loader
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"DataLoader配置 {i+1} 失败: {e}")
                
                # 检查是否是共享内存相关错误
                shared_memory_keywords = ['shared memory', 'shm', 'bus error', 'worker', 'multiprocessing']
                is_shared_memory_error = any(keyword in error_msg for keyword in shared_memory_keywords)
                
                if is_shared_memory_error:
                    print("检测到共享内存相关错误，尝试下一个配置")
                    # 如果是共享内存错误且当前配置使用多进程，跳过剩余的多进程配置
                    if config['num_workers'] > 0:
                        # 查找下一个单进程配置的索引
                        for next_idx in range(i + 1, len(configs_to_try)):
                            if configs_to_try[next_idx]['num_workers'] == 0:
                                print(f"跳过多进程配置，直接尝试单进程配置 {next_idx + 1}")
                                break
                else:
                    # 非共享内存错误，记录详细信息
                    print(f"配置失败原因: {type(e).__name__}")
                
                # 如果是最后一个配置，抛出详细错误
                if i == len(configs_to_try) - 1:
                    if is_shared_memory_error:
                        raise RuntimeError(
                            f"所有DataLoader配置都失败了，包括单进程模式。"
                            f"这可能是由于数据集或系统配置问题。最后的错误: {e}"
                        )
                    else:
                        raise RuntimeError(f"所有DataLoader配置都失败了，最后的错误: {e}")
                continue
    
    @staticmethod
    def test_dataloader_stability(loader, name="DataLoader", max_batches=5):
        """测试DataLoader的稳定性"""
        print(f"\n=== 测试 {name} 稳定性 ===")
        try:
            test_count = 0
            for batch_idx, batch_data in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                    
                # 测试数据访问 - 适配torch_geometric数据格式
                if hasattr(batch_data, 'x') and hasattr(batch_data, 'y'):
                    # torch_geometric格式
                    _ = batch_data.x.shape
                    _ = batch_data.y.shape
                    
                    # 测试数据移动到GPU（如果需要）
                    if torch.cuda.is_available():
                        try:
                            batch_gpu = batch_data.cuda(non_blocking=True)
                            # 立即释放GPU内存
                            del batch_gpu
                            torch.cuda.empty_cache()
                        except Exception as e:
                            print(f"GPU数据移动测试失败: {e}")
                            
                elif isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    # 传统格式
                    images, labels = batch_data[0], batch_data[1]
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

class PtLabelledDataset(Dataset):
    """
    通用的、基于CSV的.pt文件数据集类
    """
    def __init__(self, labels_csv_path: str, data_dirs: List[Path], 
                 split: str, suffix: str, transform=None):
        super().__init__()
        self.transform = transform
        self.data_dirs = data_dirs
        self.suffix = suffix
        self.split = split
        
        # 1. 加载并过滤CSV
        try:
            full_labels_df = pd.read_csv(labels_csv_path)
            # CSV有 'sample_id', 'label', 'split' 列
            self.labels_df = full_labels_df[full_labels_df['split'] == split].reset_index(drop=True)
        except Exception as e:
            raise FileNotFoundError(f"无法加载或处理labels CSV: {labels_csv_path}. 错误: {e}")
            
        if len(self.labels_df) == 0:
            print(f"警告: 在 {labels_csv_path} 中未找到 split='{split}' 的样本")

        # 2. 使用 BaseDataLoader 扫描所有.pt文件
        self.loaders = [BaseDataLoader(str(d), suffix=self.suffix) for d in self.data_dirs]
        self.sample_id_to_loader_map = {}
        all_found_samples = set()

        for loader in self.loaders:
            found_ids = loader.get_available_samples()
            all_found_samples.update(found_ids)
            for sample_id in found_ids:
                self.sample_id_to_loader_map[sample_id] = loader
        
        # 3. 交叉验证：只保留CSV和.pt文件都存在的样本
        csv_ids = set(self.labels_df['sample_id'])
        valid_ids = csv_ids.intersection(all_found_samples)
        
        self.labels_df = self.labels_df[self.labels_df['sample_id'].isin(valid_ids)].reset_index(drop=True)
        
        print(f"加载数据集: {self.split} - {len(self.labels_df)} 个有效样本 (CSV与.pt文件匹配成功)")

    def len(self):
        return len(self.labels_df)

    def get(self, idx):
        row = self.labels_df.iloc[idx]
        sample_id = row['sample_id']
        label = row['label']
        
        # 从map中找到对应的加载器并加载数据
        loader = self.sample_id_to_loader_map[sample_id]
        data = loader._load_sample(sample_id)
        
        if data is None:
            # 如果加载失败（例如文件损坏），返回None让DataLoader自动跳过
            print(f"警告: 加载 {sample_id} 失败，跳过。")
            return None
        
        data.y = torch.tensor(label, dtype=torch.long)
        data.filename = sample_id  # 附加文件名，供提取器使用
        
        if self.transform:
            data = self.transform(data)
            
        return data

# === 简化的数据加载器接口 ===
class DataLoaderInterface:
    """数据加载器接口 - 简化版本，移除抽象方法"""
    
    def get_train_loader(self) -> DataLoader:
        raise NotImplementedError("子类必须实现 get_train_loader 方法")
    
    def get_val_loader(self) -> DataLoader:
        raise NotImplementedError("子类必须实现 get_val_loader 方法")
    
    def get_test_loader(self) -> DataLoader:
        raise NotImplementedError("子类必须实现 get_test_loader 方法")

# === 统一的工具类 ===
class TrainingUtils:
    """训练工具类 - 统一错误处理和数据验证"""
    
    @staticmethod
    def validate_batch(batch, batch_idx: int, logger, context: str = ""):
        """统一的批次数据验证"""
        if batch is None or not hasattr(batch, 'x') or not hasattr(batch, 'y'):
            logger.warning(f"{context}批次 {batch_idx} 数据无效，跳过")
            return False
        
        if batch.x.size(0) == 0 or batch.y.size(0) == 0:
            logger.warning(f"{context}批次 {batch_idx} 包含空图，跳过")
            return False
        
        if hasattr(batch, 'edge_index') and batch.edge_index.size(1) > 0:
            max_node_idx = batch.x.size(0) - 1
            if torch.max(batch.edge_index) > max_node_idx:
                logger.warning(f"{context}批次 {batch_idx} 边索引超出节点范围，跳过")
                return False
        
        return True
    
    @staticmethod
    def validate_model_output(outputs, batch_idx: int, logger, context: str = ""):
        """统一的模型输出验证"""
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            logger.warning(f"{context}批次 {batch_idx} 模型输出包含NaN或Inf，跳过")
            return False
        return True
    
    @staticmethod
    def validate_loss(loss, batch_idx: int, logger, context: str = ""):
        """统一的损失值验证"""
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"{context}批次 {batch_idx} 损失值无效，跳过")
            return False
        return True


class UnifiedUtils:
    """统一工具类 - 合并常用的工具方法"""
    
    @staticmethod
    def create_model(model_type, config):
        """创建模型的统一接口"""
        if model_type == 'gcn':
            from common_model_components import GCNModel
            return GCNModel(config)
        elif model_type == 'gat':
            from common_model_components import GATModel
            return GATModel(config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def create_dataloader(dataset, batch_size, shuffle=False, num_workers=None, collate_fn=None, config=None):
        """创建DataLoader的统一接口"""
        if num_workers is None:
            # 如果提供了config，尝试从中获取NUM_WORKERS
            config_workers = getattr(config, 'NUM_WORKERS', None) if config else None
            num_workers = SystemManager.get_optimal_num_workers(config_workers)
        return SystemManager.create_safe_dataloader(
            dataset, batch_size, shuffle, num_workers, collate_fn
        )
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: np.ndarray = None) -> Dict[str, float]:
        """计算各种评估指标"""
        metrics = {}
        
        # 基本分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # AUC指标（需要概率）
        if y_prob is not None:
            try:
                # 检查是否为二分类问题
                unique_labels = np.unique(y_true)
                if len(unique_labels) == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_prob)
                else:
                    # 多分类问题，使用ovr策略
                    metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            except ValueError as e:
                print(f'警告: AUC计算失败: {str(e)}')
                metrics['auc'] = 0.0
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], prefix: str = ""):
        """打印评估指标"""
        for metric, value in metrics.items():
            print(f"{prefix}{metric.capitalize()}: {value:.4f}")
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: str = None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if save_path:
            plt.savefig(save_path)
        plt.show()



class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, 
                 mode: str = 'min'):
        # 参数验证
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
        if mode not in ['min', 'max']:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
            
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """检查是否应该早停"""
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, score: float) -> bool:
        """判断分数是否更好"""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

class UnifiedTrainer:
    """统一训练器"""
    
    def __init__(self, model, data_loader: DataLoaderInterface, 
                 config, model_name: str = "model"):
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.model_name = model_name
        
        # 设置设备 - 使用GPU自动选择功能
        if hasattr(config, 'DEVICE') and config.DEVICE is not None:
            self.device = config.DEVICE
        else:
            # 自动选择最佳GPU或使用CPU
            self.device = GPUManager.setup_device()
        
        print(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            GPUManager.monitor_gpu_memory(self.device)
        
        self.model.to(self.device)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0  # 添加最佳F1指标跟踪
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'train_metrics': [],
            'val_loss': [],
            'val_acc': [],
            'val_metrics': []
        }
        
        # 设置日志
        self._setup_logging()
        
        # 早停机制 - 基于F1指标
        self.early_stopping = EarlyStopping(
            patience=getattr(config, 'PATIENCE', 15),
            min_delta=getattr(config, 'MIN_DELTA', 1e-4),
            mode='max'  # F1指标越大越好
        )
        
        # 显存管理配置
        self.gradient_accumulation_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        self.memory_cleanup_frequency = getattr(config, 'MEMORY_CLEANUP_FREQUENCY', 10)
        self.max_memory_usage = getattr(config, 'MAX_MEMORY_USAGE', 0.95)  # 95%显存使用率阈值
        
        # GPU优化配置
        self.use_amp = getattr(config, 'USE_AMP', False)
        self.enable_dynamic_batch_size = getattr(config, 'ENABLE_DYNAMIC_BATCH_SIZE', False)
        self.min_batch_size = getattr(config, 'MIN_BATCH_SIZE', 4)
        self.max_batch_size = getattr(config, 'MAX_BATCH_SIZE', config.BATCH_SIZE * 2)
        self.batch_size_reduction_factor = getattr(config, 'BATCH_SIZE_REDUCTION_FACTOR', 0.75)
        
        # 梯度裁剪和数值稳定性配置
        self.gradient_clip_value = getattr(config, 'GRADIENT_CLIP_VALUE', 1.0)  # 梯度裁剪阈值
        self.enable_gradient_clipping = getattr(config, 'ENABLE_GRADIENT_CLIPPING', True)  # 启用梯度裁剪
        self.skip_nan_inf = getattr(config, 'SKIP_NAN_INF', False)  # 是否跳过NaN/Inf（用户不希望跳过）
        self.nan_inf_tolerance = getattr(config, 'NAN_INF_TOLERANCE', 1000)  # 连续NaN/Inf容忍次数（大幅增加）
        self.consecutive_nan_inf_count = 0  # 连续NaN/Inf计数器
        
        # 智能恢复机制配置
        self.auto_lr_reduction = getattr(config, 'AUTO_LR_REDUCTION', True)  # 自动学习率降低
        self.lr_reduction_factor = getattr(config, 'LR_REDUCTION_FACTOR', 0.5)  # 学习率降低因子
        self.original_lr = getattr(config, 'LEARNING_RATE', 0.001)  # 保存原始学习率
        
        # 显存安全检查
        if self.device.type == 'cuda' and self.enable_dynamic_batch_size:
            self._perform_memory_safety_check()
        
        # 混合精度训练设置
        if self.use_amp:
            self.scaler = GradScaler()
            print("混合精度训练已启用")
        else:
            self.scaler = None
        
        # 当前批大小（用于动态调整）
        self.current_batch_size = getattr(config, 'BATCH_SIZE', 32)
        
        # 初始化模型权重以提高数值稳定性
        self._initialize_model_weights()
    
    def _perform_memory_safety_check(self):
        """执行显存安全检查并调整批大小"""
        try:
            # 获取典型的输入形状（基于数据集的第一个样本）
            train_loader = self.data_loader.get_train_loader()
            sample_batch = next(iter(train_loader))
            
            # 估算输入形状
            if hasattr(sample_batch, 'x'):
                # 图数据
                avg_nodes_per_graph = sample_batch.x.size(0) // sample_batch.batch.max().item() + 1
                input_shape = (avg_nodes_per_graph, sample_batch.x.size(1))
            else:
                # 其他类型数据
                input_shape = sample_batch.shape[1:]
            
            print(f"\n=== 显存安全检查 ===")
            print(f"输入形状: {input_shape}")
            print(f"当前批大小: {self.current_batch_size}")
            
            # 检查当前批大小是否安全
            if not GPUManager.check_memory_safety(
                self.model, self.current_batch_size, input_shape, self.device
            ):
                print("当前批大小不安全，寻找最优批大小...")
                optimal_batch_size = GPUManager.find_optimal_batch_size(
                    self.model, self.current_batch_size, input_shape, self.device, self.min_batch_size
                )
                
                if optimal_batch_size < self.current_batch_size:
                    print(f"调整批大小: {self.current_batch_size} -> {optimal_batch_size}")
                    self.current_batch_size = optimal_batch_size
                    self.config.BATCH_SIZE = optimal_batch_size
                    
                    # 更新数据加载器
                    if hasattr(self.data_loader, 'update_batch_size'):
                        self.data_loader.update_batch_size(optimal_batch_size)
                        print("数据加载器已更新")
            else:
                print("当前批大小安全 ✓")
            
            print("=" * 25)
            
        except Exception as e:
            print(f"显存安全检查失败: {e}")
            print("使用保守的批大小配置")
            self.current_batch_size = self.min_batch_size
            self.config.BATCH_SIZE = self.min_batch_size
    
    def _setup_logging(self):
        """设置日志"""
        log_file = getattr(self.config, 'TRAIN_LOG', f'{self.model_name}_training.log')
        
        # 确保日志目录存在
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_gpu_memory_usage(self) -> Tuple[float, float]:
        """获取GPU显存使用情况"""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            return allocated, cached
        return 0.0, 0.0
    
    def _cleanup_memory(self):
        """清理显存 - 使用统一的内存清理功能"""
        MemoryManager.cleanup_memory(self.device, self.logger)
    
    def _check_memory_usage(self) -> bool:
        """检查显存使用率是否超过阈值"""
        if self.device.type == 'cuda':
            # 使用新的GPU内存监控功能
            GPUManager.monitor_gpu_memory(self.device)
            allocated, cached = self._get_gpu_memory_usage()
            total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            usage_ratio = cached / total_memory
            return usage_ratio > self.max_memory_usage
        return False
    
    def _adjust_batch_size_for_oom(self):
        """在遇到OOM错误时动态调整批大小"""
        if not self.enable_dynamic_batch_size:
            return False
            
        new_batch_size = int(self.current_batch_size * self.batch_size_reduction_factor)
        if new_batch_size < self.min_batch_size:
            self.logger.error(f"批大小已降至最小值 {self.min_batch_size}，无法进一步减少")
            return False
            
        self.current_batch_size = new_batch_size
        self.logger.warning(f"检测到OOM错误，将批大小调整为 {self.current_batch_size}")
        
        # 重新创建DataLoader
        try:
            self.data_loader = self._recreate_dataloader_with_new_batch_size(self.current_batch_size)
            return True
        except Exception as e:
            self.logger.error(f"重新创建DataLoader失败: {str(e)}")
            return False
    
    def _recreate_dataloader_with_new_batch_size(self, new_batch_size):
        """使用新的批大小重新创建DataLoader"""
        if hasattr(self.data_loader, 'update_batch_size'):
            self.data_loader.update_batch_size(new_batch_size)
        elif hasattr(self.data_loader, 'batch_size'):
            self.data_loader.batch_size = new_batch_size
        return self.data_loader
    
    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=getattr(self.config, 'LEARNING_RATE', 0.001),
            weight_decay=getattr(self.config, 'WEIGHT_DECAY', 1e-4)
        )
        
        # 学习率调度器
        scheduler_type = getattr(self.config, 'LR_SCHEDULER', 'step')
        if scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=getattr(self.config, 'LR_STEP_SIZE', 20),
                gamma=getattr(self.config, 'LR_GAMMA', 0.5)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=10,
                factor=0.5
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=getattr(self.config, 'EPOCHS', 100)
            )
        else:
            self.scheduler = None
    
    def train_epoch(self) -> Tuple[float, float, Dict[str, float]]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        train_loader = self.data_loader.get_train_loader()
        
        # 检查数据加载器是否为空
        if len(train_loader) == 0:
            self.logger.warning("训练数据加载器为空，跳过此epoch")
            return float('inf'), 0.0, {}
        
        # 初始化梯度累积
        accumulated_loss = 0.0
        self.optimizer.zero_grad()
        
        valid_batches = 0  # 记录有效批次数量
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # 使用统一的批次验证
                if not TrainingUtils.validate_batch(batch, batch_idx, self.logger, "训练"):
                    continue
                
                batch = batch.to(self.device)
                
                # 使用混合精度训练的前向传播
                if self.use_amp:
                    with autocast():
                        outputs = self.model(batch.x, batch.edge_index, batch.batch)
                        if not TrainingUtils.validate_model_output(outputs, batch_idx, self.logger, "训练"):
                            continue
                        loss = nn.CrossEntropyLoss()(outputs, batch.y)
                else:
                    outputs = self.model(batch.x, batch.edge_index, batch.batch)
                    if not TrainingUtils.validate_model_output(outputs, batch_idx, self.logger, "训练"):
                        continue
                    loss = nn.CrossEntropyLoss()(outputs, batch.y)
                
                # 使用新的损失处理方法
                if not self._handle_nan_inf_loss(loss, batch_idx):
                    continue
                
                # 梯度累积
                loss = loss / self.gradient_accumulation_steps
                accumulated_loss += loss.item()
                
                # 使用混合精度训练的反向传播
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # 梯度累积步骤完成时更新参数
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    
                    # 应用梯度裁剪和参数更新
                    gradient_clipping_success = True
                    
                    if self.use_amp:
                        # 先unscale梯度
                        self.scaler.unscale_(self.optimizer)
                        
                        # 应用梯度裁剪
                        gradient_clipping_success = self._apply_gradient_clipping()
                        
                        # 无论梯度裁剪是否成功，都要完成scaler的更新周期
                        if gradient_clipping_success:
                            self.scaler.step(self.optimizer)
                        else:
                            # 梯度裁剪失败时，跳过optimizer.step但仍需要update scaler
                            pass
                        
                        self.scaler.update()
                    else:
                        # 非混合精度模式
                        gradient_clipping_success = self._apply_gradient_clipping()
                        
                        if gradient_clipping_success:
                            self.optimizer.step()
                    
                    # 清零梯度（无论是否成功更新参数）
                    self.optimizer.zero_grad()
                    
                    # 如果梯度裁剪失败，跳过本次累积
                    if not gradient_clipping_success:
                        accumulated_loss = 0.0
                        continue
                    
                    # 记录累积的损失
                    total_loss += accumulated_loss
                    accumulated_loss = 0.0
                
                # 统计预测结果
                with torch.no_grad():
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
                    
                    # 安全获取正类概率
                    if probabilities.size(1) > 1:
                        all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # 正类概率
                    else:
                        # 如果只有一个类别，使用该概率
                        all_probabilities.extend(probabilities[:, 0].cpu().numpy())
                
                valid_batches += 1
                
                # 定期清理显存
                if batch_idx % self.memory_cleanup_frequency == 0:
                    self._cleanup_memory()
                    
                    # 检查显存使用情况
                    if self._check_memory_usage():
                        allocated, cached = self._get_gpu_memory_usage()
                        self.logger.warning(f'High memory usage detected - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB')
                
                # 打印进度和显存信息
                if batch_idx % getattr(self.config, 'PRINT_FREQUENCY', 10) == 0:
                    allocated, cached = self._get_gpu_memory_usage()
                    self.logger.info(f'Epoch {self.current_epoch}, Batch {batch_idx}, '
                                   f'Loss: {loss.item() * self.gradient_accumulation_steps:.4f}, '
                                   f'GPU Memory: {allocated:.2f}GB/{cached:.2f}GB')
                
            except RuntimeError as e:
                # 使用统一的CUDA错误处理
                if MemoryManager.handle_cuda_error(e, batch_idx, self.logger, "训练", 
                                                 lambda: (self._cleanup_memory(), 
                                                         self.optimizer.zero_grad())):
                    accumulated_loss = 0.0
                    
                    # 尝试动态调整批大小
                    if "out of memory" in str(e) and self._adjust_batch_size_for_oom():
                        self.logger.info("批大小调整成功，重新开始当前epoch")
                        return self.train_epoch()  # 重新开始当前epoch
                    
                    # 跳过当前batch
                    continue
                else:
                    # 其他错误直接抛出
                    raise e
        
        # 处理最后的梯度累积
        if accumulated_loss > 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += accumulated_loss
        
        if len(all_predictions) == 0 or valid_batches == 0:
            self.logger.warning("训练过程中没有有效的预测结果")
            return float('inf'), 0.0, {}
        
        avg_loss = total_loss / (len(train_loader) // self.gradient_accumulation_steps)
        
        # 计算详细指标
        try:
            train_metrics = UnifiedUtils.calculate_metrics(
                np.array(all_labels),
                np.array(all_predictions),
                np.array(all_probabilities)
            )
        except Exception as e:
            self.logger.error(f"计算训练指标时出错: {str(e)}")
            # 返回基本准确率
            accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
            train_metrics = {'accuracy': accuracy}
        
        return avg_loss, train_metrics['accuracy'] * 100, train_metrics
    
    def validate_epoch(self) -> Tuple[float, float, Dict[str, float]]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        val_loader = self.data_loader.get_val_loader()
        
        # 检查验证数据加载器是否为空
        if len(val_loader) == 0:
            self.logger.warning("验证数据加载器为空，跳过验证")
            return float('inf'), 0.0, {}
        
        valid_batches = 0  # 记录有效批次数量
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # 使用统一的批次验证
                    if not TrainingUtils.validate_batch(batch, batch_idx, self.logger, "验证"):
                        continue
                    
                    batch = batch.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(batch.x, batch.edge_index, batch.batch)
                    
                    # 使用统一的输出验证
                    if not TrainingUtils.validate_model_output(outputs, batch_idx, self.logger, "验证"):
                        continue
                    
                    loss = nn.CrossEntropyLoss()(outputs, batch.y)
                    
                    # 使用统一的损失验证
                    if not TrainingUtils.validate_loss(loss, batch_idx, self.logger, "验证"):
                        continue
                    
                    total_loss += loss.item()
                    
                    # 收集预测结果
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
                    
                    # 安全获取正类概率
                    if probabilities.size(1) > 1:
                        all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # 正类概率
                    else:
                        # 如果只有一个类别，使用该概率
                        all_probabilities.extend(probabilities[:, 0].cpu().numpy())
                    
                    valid_batches += 1
                    
                    # 定期清理显存
                    if batch_idx % self.memory_cleanup_frequency == 0:
                        self._cleanup_memory()
                        
                except RuntimeError as e:
                    # 使用统一的CUDA错误处理
                    if MemoryManager.handle_cuda_error(e, batch_idx, self.logger, "验证", 
                                                     self._cleanup_memory):
                        # 跳过当前batch
                        continue
                    else:
                        # 其他错误直接抛出
                        raise e
        
        if len(all_predictions) == 0 or valid_batches == 0:
            self.logger.warning("验证过程中没有有效的预测结果")
            return float('inf'), 0.0, {}
        
        avg_loss = total_loss / valid_batches
        
        # 计算详细指标
        try:
            metrics = UnifiedUtils.calculate_metrics(
                np.array(all_labels),
                np.array(all_predictions),
                np.array(all_probabilities)
            )
        except Exception as e:
            self.logger.error(f"计算验证指标时出错: {str(e)}")
            # 返回基本准确率
            accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
            metrics = {'accuracy': accuracy}
        
        return avg_loss, metrics['accuracy'] * 100, metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,  # 添加最佳F1指标
            'training_history': self.training_history
        }
        
        # 保存最新检查点
        checkpoint_dir = getattr(self.config, 'CHECKPOINT_DIR', Path('checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'{self.model_name}_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = getattr(self.config, 'BEST_MODEL_PATH', checkpoint_dir / f'best_{self.model_name}.pt')
            # 确保路径是Path对象
            if isinstance(best_path, str):
                best_path = Path(best_path)
            # 确保父目录存在
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, best_path)
            self.logger.info(f'保存最佳模型到: {best_path}')
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f'检查点文件不存在: {checkpoint_path}')
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)  # 兼容旧检查点
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f'从检查点恢复训练: epoch {self.current_epoch}')
    
    def train(self, resume_from: str = None):
        """完整训练流程"""
        self.logger.info(f'开始训练 {self.model_name} 模型')
        self.logger.info(f'设备: {self.device}')
        self.logger.info(f'模型参数数量: {sum(p.numel() for p in self.model.parameters())}')
        
        # 设置优化器
        self.setup_optimizer()
        
        # 恢复训练
        if resume_from:
            self.load_checkpoint(resume_from)
        
        epochs = getattr(self.config, 'EPOCHS', 100)
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            try:
                # 记录显存使用情况
                allocated, cached = self._get_gpu_memory_usage()
                self.logger.info(f'Epoch {epoch}: GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB')
                
                # 训练
                train_loss, train_acc, train_metrics = self.train_epoch()
                
                # 验证
                if epoch % getattr(self.config, 'VAL_FREQUENCY', 1) == 0:
                    val_loss, val_acc, val_metrics = self.validate_epoch()
                    
                    # 记录历史
                    self.training_history['train_loss'].append(train_loss)
                    self.training_history['train_acc'].append(train_acc)
                    self.training_history['train_metrics'].append(train_metrics)
                    self.training_history['val_loss'].append(val_loss)
                    self.training_history['val_acc'].append(val_acc)
                    self.training_history['val_metrics'].append(val_metrics)
                    
                    # 更新学习率
                    if self.scheduler:
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()
                    
                    # 检查是否是最佳模型 - 基于F1指标
                    val_f1 = val_metrics.get('f1', 0.0)
                    is_best = val_f1 > self.best_val_f1
                    if is_best:
                        self.best_val_f1 = val_f1
                        self.best_val_acc = val_acc
                        self.best_val_loss = val_loss
                        self.logger.info(f'新的最佳模型! F1: {val_f1:.4f}')
                    
                    # 保存检查点
                    if epoch % getattr(self.config, 'SAVE_FREQUENCY', 10) == 0 or is_best:
                        self.save_checkpoint(is_best)
                    
                    # 早停检查 - 基于验证集F1指标
                    if self.early_stopping(val_f1):
                        self.logger.info(f'早停触发，在epoch {epoch}停止训练 (基于F1指标)')
                        break
                    
                    # 打印详细进度信息
                    epoch_time = time.time() - start_time
                    
                    # 获取详细指标用于显示
                    train_pre = train_metrics.get('precision', 0.0) * 100
                    train_r1 = train_metrics.get('recall', 0.0) * 100
                    train_f1 = train_metrics.get('f1', 0.0) * 100
                    
                    val_pre = val_metrics.get('precision', 0.0) * 100
                    val_r1 = val_metrics.get('recall', 0.0) * 100
                    val_f1 = val_metrics.get('f1', 0.0) * 100
                    
                    self.logger.info(
                        f'Epoch {epoch}/{epochs-1} - Time: {epoch_time:.2f}s\n'
                        f'  Train - Loss: {train_loss:.4f}, ACC: {train_acc:.2f}%, PRE: {train_pre:.2f}%, R1: {train_r1:.2f}%, F1: {train_f1:.2f}%\n'
                        f'  Val   - Loss: {val_loss:.4f}, ACC: {val_acc:.2f}%, PRE: {val_pre:.2f}%, R1: {val_r1:.2f}%, F1: {val_f1:.2f}%'
                    )
                
                # 重置连续错误计数
                consecutive_errors = 0
                
            except RuntimeError as e:
                if "out of memory" in str(e) or "CUDA error" in str(e):
                    consecutive_errors += 1
                    self.logger.error(f'CUDA error at epoch {epoch}: {str(e)}')
                    self.logger.error(f'Consecutive errors: {consecutive_errors}/{max_consecutive_errors}')
                    
                    # 清理显存
                    self._cleanup_memory()
                    
                    # 如果连续错误太多，停止训练
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error(f'Too many consecutive CUDA errors ({consecutive_errors}). Stopping training.')
                        break
                    
                    # 尝试动态调整批大小
                    if hasattr(self.config, 'BATCH_SIZE') and self.config.BATCH_SIZE > 1:
                        old_batch_size = self.config.BATCH_SIZE
                        self.config.BATCH_SIZE = max(1, self.config.BATCH_SIZE // 2)
                        self.logger.warning(f'Reducing batch size from {old_batch_size} to {self.config.BATCH_SIZE}')
                        
                        # 重新创建数据加载器
                        try:
                            # 保持原始数据加载器的类型和参数
                            if hasattr(self.data_loader, 'config') and hasattr(self.data_loader, 'train_loader'):
                                # 更新现有数据加载器的批次大小
                                old_train_loader = self.data_loader.train_loader
                                old_val_loader = self.data_loader.val_loader
                                
                                # 重新创建DataLoader with new batch size
                                self.data_loader.train_loader = DataLoader(
                                    old_train_loader.dataset, 
                                    batch_size=self.config.BATCH_SIZE, 
                                    shuffle=True,
                                    num_workers=self.config.NUM_WORKERS, 
                                    pin_memory=self.config.PIN_MEMORY
                                )
                                self.data_loader.val_loader = DataLoader(
                                    old_val_loader.dataset, 
                                    batch_size=self.config.BATCH_SIZE, 
                                    shuffle=False,
                                    num_workers=self.config.NUM_WORKERS, 
                                    pin_memory=self.config.PIN_MEMORY
                                )
                            else:
                                self.logger.warning('Cannot recreate data loader: incompatible type')
                        except Exception as loader_e:
                            self.logger.error(f'Failed to recreate data loader: {str(loader_e)}')
                            break
                    
                    # 跳过当前epoch
                    continue
                else:
                    # 其他错误直接抛出
                    raise e
            
            except Exception as e:
                self.logger.error(f'Unexpected error at epoch {epoch}: {str(e)}')
                consecutive_errors += 1
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(f'Too many consecutive errors ({consecutive_errors}). Stopping training.')
                    break
                
                continue
        
        # 保存训练历史
        self._save_training_history()
        self.logger.info('训练完成')
    
    def _save_training_history(self):
        """保存训练历史"""
        history_path = getattr(self.config, 'CHECKPOINT_DIR', Path('checkpoints')) / f'{self.model_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def evaluate(self, test_loader=None):
        """评估模型"""
        self.logger.info('开始模型评估')
        
        if test_loader is None:
            test_loader = self.data_loader.get_test_loader()
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                # 前向传播 - 移除.is_dotnet逻辑
                outputs = self.model(batch.x, batch.edge_index, batch.batch)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                
                # 安全获取正类概率
                if probabilities.size(1) > 1:
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # 正类概率
                else:
                    # 如果只有一个类别，使用该概率
                    all_probabilities.extend(probabilities[:, 0].cpu().numpy())
        
        # 计算指标
        metrics = UnifiedUtils.calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        self.logger.info('测试集评估结果:')
        self.logger.info(f'  ACC: {metrics["accuracy"]*100:.2f}%')
        self.logger.info(f'  PRE: {metrics["precision"]*100:.2f}%')
        self.logger.info(f'  R1:  {metrics["recall"]*100:.2f}%')
        self.logger.info(f'  F1:  {metrics["f1"]*100:.2f}%')
        if 'auc' in metrics:
            self.logger.info(f'  AUC: {metrics["auc"]:.4f}')
        
        # 也使用原有的打印方法作为备份
        UnifiedUtils.print_metrics(metrics, "Test ")
        
        # 绘制混淆矩阵
        cm_path = getattr(self.config, 'CHECKPOINT_DIR', Path('checkpoints')) / f'{self.model_name}_confusion_matrix.png'
        UnifiedUtils.plot_confusion_matrix(
            np.array(all_labels),
            np.array(all_predictions),
            str(cm_path)
        )
        
        return metrics

    def _initialize_model_weights(self):
        """初始化模型权重以提高数值稳定性"""
        try:
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    # Xavier初始化线性层
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
                    # Kaiming初始化卷积层
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                    # 批归一化层初始化
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)
            
            print("模型权重初始化完成")
        except Exception as e:
            self.logger.warning(f"模型权重初始化失败: {e}")
    
    def _apply_gradient_clipping(self):
        """应用梯度裁剪"""
        if self.enable_gradient_clipping:
            # 计算梯度范数
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.gradient_clip_value
            )
            
            # 检查梯度是否包含NaN或Inf
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                self.logger.warning(f"检测到异常梯度范数: {total_norm}")
                # 清零梯度
                self.optimizer.zero_grad()
                return False
            
            # 如果梯度范数过大，记录警告
            if total_norm > self.gradient_clip_value * 2:
                self.logger.warning(f"梯度范数较大: {total_norm:.4f}")
            
            return True
        return True
    
    def _handle_nan_inf_loss(self, loss, batch_idx):
        """处理NaN/Inf损失值 - 改进版本，支持更高容忍度和智能恢复"""
        if torch.isnan(loss) or torch.isinf(loss):
            self.consecutive_nan_inf_count += 1
            
            # 每10次异常打印一次详细信息，避免日志过多
            if self.consecutive_nan_inf_count % 10 == 1 or self.consecutive_nan_inf_count <= 10:
                self.logger.warning(f"批次 {batch_idx} 损失值异常: {loss.item()}, 连续异常次数: {self.consecutive_nan_inf_count}")
            
            # 智能恢复策略
            if self.consecutive_nan_inf_count % 50 == 0:
                # 每50次异常尝试重新初始化优化器状态
                self.logger.info(f"连续异常达到 {self.consecutive_nan_inf_count} 次，尝试重置优化器状态")
                self.optimizer.zero_grad()
                # 重置优化器的内部状态
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                        # 清除Adam优化器的动量缓存
                        param_state = self.optimizer.state.get(p, {})
                        if 'exp_avg' in param_state:
                            param_state['exp_avg'].zero_()
                        if 'exp_avg_sq' in param_state:
                            param_state['exp_avg_sq'].zero_()
            
            # 自动学习率调整
            if self.auto_lr_reduction and self.consecutive_nan_inf_count % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                new_lr = current_lr * self.lr_reduction_factor
                if new_lr > self.original_lr * 0.001:  # 最低不低于原始学习率的0.1%
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    self.logger.info(f"自动降低学习率: {current_lr:.6f} -> {new_lr:.6f}")
                else:
                    self.logger.warning(f"学习率已达到最低限制: {current_lr:.6f}")
            
            if not self.skip_nan_inf:
                # 用户不希望跳过，但大幅增加容忍度
                if self.consecutive_nan_inf_count <= self.nan_inf_tolerance:
                    # 清零梯度并跳过当前批次
                    self.optimizer.zero_grad()
                    return False
                else:
                    # 超过容忍次数，但不立即停止，而是警告并继续
                    if self.consecutive_nan_inf_count % 100 == 0:
                        self.logger.error(f"连续 {self.consecutive_nan_inf_count} 次遇到NaN/Inf损失，训练可能不稳定，但继续尝试")
                    self.optimizer.zero_grad()
                    return False
            else:
                # 用户允许跳过
                self.optimizer.zero_grad()
                return False
        else:
            # 正常损失，重置连续异常计数
            if self.consecutive_nan_inf_count > 0:
                self.logger.info(f"损失恢复正常，重置异常计数器 (之前连续异常 {self.consecutive_nan_inf_count} 次)")
                self.consecutive_nan_inf_count = 0
                
                # 如果启用了自动学习率调整，考虑恢复学习率
                if self.auto_lr_reduction:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if current_lr < self.original_lr * 0.8:  # 如果学习率被显著降低
                        # 缓慢恢复学习率
                        new_lr = min(current_lr * 1.1, self.original_lr)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        self.logger.info(f"损失恢复，缓慢提升学习率: {current_lr:.6f} -> {new_lr:.6f}")
            return True


def create_unified_trainer(model_type: str, config_path: str = None, 
                          labels_csv_path: str = None, enable_gpu_optimization: bool = False) -> UnifiedTrainer:
    """
    创建统一训练器的工厂函数
    
    Args:
        model_type: 模型类型 ('cfg' 或 'fcg')
        config_path: 配置文件路径 (可选)
        labels_csv_path: 标签CSV文件路径 (可选)
        enable_gpu_optimization: 是否启用GPU优化 (默认False，设置为True可大幅提升GPU利用率)
    
    Returns:
        UnifiedTrainer: 配置好的训练器实例
        
    使用示例:
        # 基础使用 (保持原有性能)
        trainer = create_unified_trainer('cfg')
        
        # 启用GPU优化 (推荐，可提升GPU利用率到80%+)
        trainer = create_unified_trainer('cfg', enable_gpu_optimization=True)
    """
    
    # 显示GPU信息
    print("=== GPU信息 ===")
    gpu_info = GPUManager.get_gpu_memory_usage()
    if gpu_info:
        for info in gpu_info:
            print(f"GPU {info['gpu_id']}: {info['name']}")
            print(f"  总显存: {info['total_memory']:.1f} GB")
            print(f"  已用显存: {info['used_memory']:.1f} GB")
            print(f"  可用显存: {info['free_memory']:.1f} GB")
            print(f"  使用率: {info['memory_usage']:.1f}%")
    else:
        print("未检测到可用的GPU")
    print("=" * 20)
    
    # 根据模型类型选择配置
    if model_type == 'cfg':
        config = CFGConfig()
        data_dirs = [config.BENIGN_CFG_DIR, config.MALWARE_CFG_DIR]
        suffix = config.CFG_SUFFIX
    elif model_type == 'fcg':
        config = FCGConfig()
        data_dirs = [config.BENIGN_FCG_DIR, config.MALWARE_FCG_DIR]
        suffix = config.FCG_SUFFIX
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 应用GPU优化配置
    if enable_gpu_optimization:
        print(f"\n=== GPU优化已启用 ===")
        # 获取GPU信息用于显示
        gpu_info = GPUManager.get_gpu_memory_usage()
        total_memory = 0
        if gpu_info:
            total_memory = max([info['total_memory'] for info in gpu_info])
        
        original_batch_size = config.BATCH_SIZE
        original_num_workers = config.NUM_WORKERS
        
        # 直接使用各模型的固定配置（不再根据显存自适应调整）
        # get_adaptive_batch_config现在返回固定配置
        config.BATCH_SIZE, config.NUM_WORKERS = get_adaptive_batch_config(
            model_type, total_memory, original_batch_size
        )
        
        # 启用GPU优化特性
        config.USE_AMP = torch.cuda.is_available() and total_memory >= 8
        config.ENABLE_DYNAMIC_BATCH_SIZE = False  # 禁用动态批大小调整
        config.PIN_MEMORY = True
        
        # 使用固定的批大小边界（基于配置的固定值）
        config.MIN_BATCH_SIZE = max(2, config.BATCH_SIZE // 2)  # 最小批大小
        config.MAX_BATCH_SIZE = config.BATCH_SIZE  # 最大批大小等于固定批大小
        config.BATCH_SIZE_REDUCTION_FACTOR = 0.75  # OOM时的缩减因子
        
        print(f"模型类型: {model_type.upper()}")
        print(f"GPU显存: {total_memory:.1f}GB")
        print(f"使用固定批大小: {config.BATCH_SIZE} (原始: {original_batch_size})")
        print(f"使用固定Worker数量: {config.NUM_WORKERS} (原始: {original_num_workers})")
        print(f"批大小范围: {config.MIN_BATCH_SIZE} - {config.MAX_BATCH_SIZE}")
        print(f"混合精度训练: {getattr(config, 'USE_AMP', False)}")
        print(f"动态批大小调整: {getattr(config, 'ENABLE_DYNAMIC_BATCH_SIZE', False)} (已禁用)")
        print("=" * 25)
    else:
        print("\n💡 提示: 添加 enable_gpu_optimization=True 参数可启用GPU优化特性")
        print("   示例: create_unified_trainer('cfg', enable_gpu_optimization=True)")
        print("=" * 60)

    if labels_csv_path:
        config.LABELS_CSV = Path(labels_csv_path)

    config.create_directories()
    model = create_model_from_config(config, model_type='classifier')

    # 创建数据加载器适配器
    class PtDataLoaderAdapter(DataLoaderInterface):
        def __init__(self, config_obj, data_dirs, suffix):
            self.config = config_obj
            self.data_dirs = data_dirs
            self.suffix = suffix
            csv_path = str(self.config.LABELS_CSV)
            
            self.train_dataset = PtLabelledDataset(csv_path, data_dirs, 'train', suffix)
            self.val_dataset = PtLabelledDataset(csv_path, data_dirs, 'val', suffix)
            self.test_dataset = PtLabelledDataset(csv_path, data_dirs, 'test', suffix)
            
            # 当前批大小
            self.current_batch_size = self.config.BATCH_SIZE
            
            # 创建初始DataLoader
            self._create_dataloaders()
            
        def _create_dataloaders(self):
            """创建或重新创建DataLoader"""
            # 使用增强的DataLoader创建功能
            print(f"\n=== 初始化增强DataLoader (批大小: {self.current_batch_size}) ===")
            
            # 优先使用配置文件中的NUM_WORKERS设置
            if hasattr(self.config, 'NUM_WORKERS') and self.config.NUM_WORKERS is not None:
                optimal_workers = self.config.NUM_WORKERS
                print(f"使用配置文件中的worker数量: {optimal_workers}")
            else:
                # 如果配置文件中没有设置，则使用动态检测
                config_workers = getattr(self.config, 'NUM_WORKERS', None)
                optimal_workers = SystemManager.get_optimal_num_workers(config_workers)
                print(f"使用动态检测的worker数量: {optimal_workers}")
            
            # 创建训练DataLoader（需要shuffle）
            print("创建训练DataLoader...")
            self.train_loader = SystemManager.create_safe_dataloader(
                dataset=self.train_dataset,
                batch_size=self.current_batch_size,
                shuffle=True,
                optimal_workers=optimal_workers
            )
            
            # 创建验证DataLoader
            print("创建验证DataLoader...")
            self.val_loader = SystemManager.create_safe_dataloader(
                dataset=self.val_dataset,
                batch_size=self.current_batch_size,
                shuffle=False,
                optimal_workers=optimal_workers
            )
            
            # 创建测试DataLoader
            print("创建测试DataLoader...")
            self.test_loader = SystemManager.create_safe_dataloader(
                dataset=self.test_dataset,
                batch_size=self.current_batch_size,
                shuffle=False,
                optimal_workers=optimal_workers
            )
            
            # 测试DataLoader稳定性
            print("\n=== 测试DataLoader稳定性 ===")
            if not SystemManager.test_dataloader_stability(self.train_loader, "训练DataLoader", max_batches=3):
                raise RuntimeError("训练DataLoader稳定性测试失败")
            if not SystemManager.test_dataloader_stability(self.val_loader, "验证DataLoader", max_batches=2):
                raise RuntimeError("验证DataLoader稳定性测试失败")
            if not SystemManager.test_dataloader_stability(self.test_loader, "测试DataLoader", max_batches=2):
                raise RuntimeError("测试DataLoader稳定性测试失败")
            
            print("所有DataLoader创建成功并通过稳定性测试")
            
        def update_batch_size(self, new_batch_size):
            """更新批大小并重新创建DataLoader"""
            self.current_batch_size = new_batch_size
            self._create_dataloaders()

        def get_train_loader(self): return self.train_loader
        def get_val_loader(self): return self.val_loader
        def get_test_loader(self): return self.test_loader

    data_loader = PtDataLoaderAdapter(config, data_dirs, suffix)
    return UnifiedTrainer(model, data_loader, config, f"{model_type}_model")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一训练框架')
    parser.add_argument('--model_type', type=str, choices=['cfg', 'fcg'], 
                       required=True, help='模型类型')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--labels_csv', type=str, default=None,
                       help='labels.csv文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='仅进行评估，不训练')
    parser.add_argument('--gpu_optimization', action='store_true',
                       help='启用GPU优化 (推荐，可大幅提升GPU利用率)')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = create_unified_trainer(
        model_type=args.model_type,
        config_path=args.config,
        labels_csv_path=args.labels_csv,
        enable_gpu_optimization=args.gpu_optimization
    )
    
    if args.evaluate_only:
        # 仅评估
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.evaluate()
    else:
        # 训练
        trainer.train(resume_from=args.resume)
        
        # 训练完成后评估
        trainer.evaluate()

if __name__ == "__main__":
    main()