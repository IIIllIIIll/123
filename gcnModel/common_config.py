#!/usr/bin/env python3
"""
公共配置基类
提取CFG和FCG训练配置中的共同部分
包含GPU和显存管理功能
"""

import gc
import os
import subprocess
from pathlib import Path
from typing import Tuple

# PyTorch相关导入
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，GPU功能将不可用")


# === GPU和显存管理器 ===
class GPUManager:
    """GPU和显存管理器 - 统一管理GPU相关功能"""
    
    @staticmethod
    def estimate_memory_usage(model, batch_size: int, input_shape: tuple, device) -> float:
        """
        预估模型在给定批大小下的显存使用量
        
        Args:
            model: PyTorch模型
            batch_size: 批大小
            input_shape: 输入数据形状 (不包括batch维度)
            device: 计算设备
            
        Returns:
            float: 预估的显存使用量(GB)
        """
        if not TORCH_AVAILABLE:
            return 0.0
            
        try:
            # 清理显存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated(device) / 1024**3
            else:
                return 0.0  # CPU模式不需要显存预估
            
            model.eval()
            
            # 创建虚拟输入数据进行测试
            with torch.no_grad():
                # 对于图数据，创建简化的测试输入
                if len(input_shape) == 2:  # 节点特征 (num_nodes, num_features)
                    test_x = torch.randn(input_shape[0] * batch_size, input_shape[1], device=device)
                    # 创建简单的边索引
                    num_edges = min(input_shape[0] * 2, 1000)  # 限制边数量避免过大
                    test_edge_index = torch.randint(0, input_shape[0], (2, num_edges), device=device)
                    # 创建批次索引
                    test_batch = torch.repeat_interleave(torch.arange(batch_size, device=device), input_shape[0])
                    
                    # 前向传播测试
                    _ = model(test_x, test_edge_index, test_batch)
                else:
                    # 其他类型的输入
                    test_input = torch.randn(batch_size, *input_shape, device=device)
                    _ = model(test_input)
                
                # 计算显存使用量
                peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
                estimated_usage = peak_memory - initial_memory
                
                # 重置显存统计
                torch.cuda.reset_peak_memory_stats(device)
                
                return max(0.0, estimated_usage)
                
        except Exception as e:
            print(f"显存预估失败: {e}")
            # 返回保守估计值
            return batch_size * 0.1  # 每个样本约100MB
    
    @staticmethod
    def check_memory_safety(model, batch_size: int, input_shape: tuple, device, 
                          safety_margin: float = 0.2) -> bool:
        """
        检查给定批大小是否安全（不会导致OOM）
        
        Args:
            model: PyTorch模型
            batch_size: 批大小
            input_shape: 输入数据形状
            device: 计算设备
            safety_margin: 安全边界（预留显存比例）
            
        Returns:
            bool: 是否安全
        """
        if not TORCH_AVAILABLE or device.type != 'cuda':
            return True  # CPU模式总是安全的
        
        try:
            # 获取GPU显存信息
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            current_memory = torch.cuda.memory_allocated(device) / 1024**3
            available_memory = total_memory - current_memory
            
            # 预估显存使用量
            estimated_usage = GPUManager.estimate_memory_usage(model, batch_size, input_shape, device)
            
            # 考虑安全边界
            required_memory = estimated_usage * (1 + safety_margin)
            
            is_safe = required_memory <= available_memory
            
            if not is_safe:
                print(f"显存安全检查失败:")
                print(f"  总显存: {total_memory:.2f}GB")
                print(f"  已用显存: {current_memory:.2f}GB") 
                print(f"  可用显存: {available_memory:.2f}GB")
                print(f"  预估需求: {estimated_usage:.2f}GB")
                print(f"  安全需求: {required_memory:.2f}GB")
            
            return is_safe
            
        except Exception as e:
            print(f"显存安全检查失败: {e}")
            return False  # 出错时采用保守策略
    
    @staticmethod
    def find_optimal_batch_size(model, max_batch_size: int, input_shape: tuple, device,
                              min_batch_size: int = 4) -> int:
        """
        通过二分搜索找到最优批大小
        
        Args:
            model: PyTorch模型
            max_batch_size: 最大批大小
            input_shape: 输入数据形状
            device: 计算设备
            min_batch_size: 最小批大小
            
        Returns:
            int: 最优批大小
        """
        if not TORCH_AVAILABLE or device.type != 'cuda':
            return max_batch_size  # CPU模式直接返回最大值
        
        left, right = min_batch_size, max_batch_size
        optimal_batch_size = min_batch_size
        
        print(f"开始二分搜索最优批大小 (范围: {left}-{right})")
        
        while left <= right:
            mid = (left + right) // 2
            
            if GPUManager.check_memory_safety(model, mid, input_shape, device):
                optimal_batch_size = mid
                left = mid + 1
                print(f"  批大小 {mid}: 安全 ✓")
            else:
                right = mid - 1
                print(f"  批大小 {mid}: 不安全 ✗")
        
        print(f"找到最优批大小: {optimal_batch_size}")
        return optimal_batch_size

    @staticmethod
    def get_gpu_memory_usage():
        """获取所有GPU的显存使用情况"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    gpu_id = int(parts[0])
                    gpu_name = parts[1]
                    memory_used = int(parts[2])  # MB
                    memory_total = int(parts[3])  # MB
                    gpu_util = int(parts[4])
                    memory_usage_percent = (memory_used / memory_total) * 100
                    
                    gpu_info.append({
                        'gpu_id': gpu_id,
                        'name': gpu_name,
                        'memory_used': memory_used,
                        'memory_total': memory_total,
                        'memory_free': memory_total - memory_used,
                        'memory_usage_percent': memory_usage_percent,
                        'gpu_utilization': gpu_util,
                        # Add aliases for compatibility
                        'used_memory': memory_used / 1024.0,  # Convert to GB
                        'total_memory': memory_total / 1024.0,  # Convert to GB
                        'free_memory': (memory_total - memory_used) / 1024.0,  # Convert to GB
                        'memory_usage': memory_usage_percent
                    })
            
            return gpu_info
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            print(f"警告: 无法获取GPU信息: {e}")
            return []
    
    @staticmethod
    def find_best_gpu(min_free_memory_gb=2, max_memory_usage_percent=80):
        """找到最适合的GPU"""
        gpu_info = GPUManager.get_gpu_memory_usage()
        
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
    
    @staticmethod
    def setup_device(gpu_id=None):
        """设置训练设备"""
        if not TORCH_AVAILABLE:
            print("PyTorch不可用，无法设置设备")
            return None
            
        if not torch.cuda.is_available():
            print("CUDA不可用，使用CPU")
            return torch.device("cpu")
        
        if gpu_id is None:
            gpu_id = GPUManager.find_best_gpu()
        
        if gpu_id is not None:
            device = torch.device(f"cuda:{gpu_id}")
            torch.cuda.set_device(gpu_id)
            print(f"使用GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            GPUManager.cleanup_gpu_memory(device)
            return device
        else:
            device = torch.device("cuda")
            print(f"使用默认GPU: {torch.cuda.get_device_name()}")
            return device
    
    @staticmethod
    def monitor_gpu_memory(device):
        """监控GPU显存使用情况"""
        if not TORCH_AVAILABLE or device is None:
            return
            
        if device.type == 'cuda':
            gpu_id = device.index if device.index is not None else 0
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3   # GB
            print(f"GPU {gpu_id} 显存使用: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
    
    @staticmethod
    def cleanup_gpu_memory(device):
        """清理GPU显存"""
        if not TORCH_AVAILABLE or device is None:
            return
            
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print("已清理GPU缓存")


class MemoryManager:
    """内存管理器 - 统一管理内存相关功能"""
    
    @staticmethod
    def cleanup_memory(device, logger=None):
        """统一的内存清理"""
        if not TORCH_AVAILABLE:
            gc.collect()
            if logger:
                logger.debug("已清理系统内存")
            return
            
        if device and device.type == 'cuda':
            torch.cuda.empty_cache()
            if logger:
                logger.debug("已清理GPU缓存")
        else:
            gc.collect()
            if logger:
                logger.debug("已清理系统内存")
    
    @staticmethod
    def handle_cuda_error(e: Exception, batch_idx: int, logger, context: str = "", 
                         cleanup_callback=None) -> bool:
        """统一的CUDA错误处理"""
        error_str = str(e).lower()
        if "out of memory" in error_str or "cuda error" in error_str:
            if logger:
                logger.error(f'{context}CUDA error at batch {batch_idx}: {str(e)}')
            else:
                print(f'{context}CUDA error at batch {batch_idx}: {str(e)}')
            if cleanup_callback:
                cleanup_callback()
            return True
        return False


def get_adaptive_batch_config(model_type: str, total_memory_gb: float, original_batch_size: int) -> tuple:
    """
    返回固定的批大小和worker数量配置（不再根据显存自适应调整）
    
    Args:
        model_type: 模型类型 ('cfg' 或 'fcg')
        total_memory_gb: GPU总显存(GB) - 此参数保留兼容性但不使用
        original_batch_size: 原始批大小 - 此参数保留兼容性但不使用
        
    Returns:
        tuple: (固定批大小, worker数量)
    """
    # 直接返回各模型的固定配置
    if model_type.lower() == 'cfg':
        return CFGConfig.BATCH_SIZE, CFGConfig.NUM_WORKERS
    elif model_type.lower() == 'fcg':
        return FCGConfig.BATCH_SIZE, FCGConfig.NUM_WORKERS
    else:
        # 默认配置
        return 8, 1


class BaseConfig:
    """基础配置类，包含通用配置项"""
    
    # # ========== 通用路径配置 ==========
    # PROJECT_ROOT = Path("/mnt/data1_l20_raid5disk/lbq_dataset")
    
    # # 通用数据路径
    # LABELS_CSV = PROJECT_ROOT / "output" / "labels.csv"
    
    # # 模型保存路径 - 统一放在output目录下
    # MODEL_DIR = PROJECT_ROOT / "output"
    
    # # 日志路径
    # LOG_DIR = PROJECT_ROOT / "output"

    # ========== 通用路径配置 ==========
    PROJECT_ROOT = Path("/mnt/lbq")
    
    # 通用数据路径
    LABELS_CSV = PROJECT_ROOT / "output/labels.csv"
    
    # 模型保存路径 - 统一放在output目录下
    MODEL_DIR = PROJECT_ROOT / "output/model"
    
    # 日志路径
    LOG_DIR = PROJECT_ROOT / "output/log"
    
    # ========== FCG特定路径和文件名 ==========
    BENIGN_FCG_DIR = PROJECT_ROOT / "dataset" / "benign_fcg"
    MALWARE_FCG_DIR = PROJECT_ROOT / "dataset" / "malware_fcg"
    FCG_SUFFIX = "_fcg_graph.pt"   # FCG文件名后缀
    
    # ========== CFG特定路径和文件名 ==========
    BENIGN_CFG_DIR = PROJECT_ROOT / "dataset" / "benign_cfg"
    MALWARE_CFG_DIR = PROJECT_ROOT / "dataset" / "malware_cfg"
    CFG_SUFFIX = "_ida_analysis_gcn_data.pt"  # CFG文件名后缀
    
    # ========== 通用模型超参数 ==========
    # GCN通用参数
    HIDDEN_DIM = 128        # 隐藏层维度
    NUM_GCN_LAYERS = 3      # GCN层数
    DROPOUT = 0.3           # Dropout率
    NUM_CLASSES = 2         # 分类数量（良性/恶意）
    
    # ========== 通用训练超参数 ==========
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4     # L2正则化
    
    # 训练参数
    EPOCHS = 100            # 训练轮数
    BATCH_SIZE = 64         # 批大小
    
    # 早停参数
    PATIENCE = 15           # 早停耐心值
    MIN_DELTA = 1e-4        # 最小改善阈值
    
    # 学习率调度器参数
    LR_SCHEDULER = 'step'   # 学习率调度器类型: 'step', 'plateau', 'cosine'
    LR_STEP_SIZE = 20       # 学习率衰减步长
    LR_GAMMA = 0.5          # 学习率衰减因子
    
    # ========== 数据加载参数 ==========
    NUM_WORKERS = 1         # 数据加载进程数
    PIN_MEMORY = True       # 是否使用pin_memory
    
    # ========== 设备配置 ==========
    try:
        import torch
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except ImportError:
        DEVICE = 'cpu'  # 默认CPU
    
    # ========== 训练监控参数 ==========
    VAL_FREQUENCY = 1       # 每多少个epoch验证一次
    SAVE_FREQUENCY = 10     # 每多少个epoch保存一次checkpoint
    PRINT_FREQUENCY = 10    # 每多少个batch打印一次训练信息
    
    # ========== 评估指标 ==========
    METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.MODEL_DIR,
            cls.LOG_DIR,
        ]
        
        # 为CFG和FCG模型创建专用目录
        if hasattr(cls, 'CHECKPOINT_DIR'):
            directories.append(cls.CHECKPOINT_DIR)
        if hasattr(cls, 'BEST_MODEL_PATH'):
            directories.append(cls.BEST_MODEL_PATH.parent)
        if hasattr(cls, 'TRAIN_LOG'):
            directories.append(cls.TRAIN_LOG.parent)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {directory}")
    
    @classmethod
    def get_config_dict(cls):
        """获取配置字典"""
        config_dict = {}
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and not callable(getattr(cls, attr_name)):
                config_dict[attr_name] = getattr(cls, attr_name)
        return config_dict
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 50)
        print(f"{cls.__name__} 配置信息:")
        print("=" * 50)
        
        config_dict = cls.get_config_dict()
        for key, value in config_dict.items():
            if not key.startswith('_'):
                print(f"{key:25}: {value}")
        print("=" * 50)


# class BaseProductionConfig(BaseConfig):
#     """生产环境基础配置"""
    
#     # 生产环境参数调整
#     EPOCHS = 200            # 增加训练轮数
#     BATCH_SIZE = 64         # 增大批大小
#     PATIENCE = 20           # 增加早停耐心值
#     LEARNING_RATE = 0.0005  # 降低学习率
    
#     # 模型参数调整
#     HIDDEN_DIM = 256
#     NUM_GCN_LAYERS = 4


# class BaseDebugConfig(BaseConfig):
#     """调试环境基础配置"""
    
#     # 调试参数
#     DEBUG_MODE = True
#     MAX_SAMPLES = 100       # 限制样本数量
#     EPOCHS = 5              # 减少训练轮数
#     BATCH_SIZE = 8          # 减小批大小
#     PATIENCE = 3            # 减少早停耐心值
#     PRINT_FREQUENCY = 1     # 增加打印频率
#     VAL_FREQUENCY = 1       # 每个epoch都验证


# def get_base_config(mode: str = 'default'):
#     """
#     根据模式返回对应的配置类
    
#     Args:
#         mode: 配置模式 ('default', 'production', 'debug')
    
#     Returns:
#         对应的配置类
#     """
#     if mode == 'production':
#         return BaseProductionConfig
#     elif mode == 'debug':
#         return BaseDebugConfig
#     else:
#         return BaseConfig


class CFGConfig(BaseConfig):
    """CFG模型专用配置"""
    INPUT_DIM = 100
    EMBEDDING_DIM = 128
    
    # CFG模型固定训练参数
    BATCH_SIZE = 32          # CFG模型固定批大小
    LEARNING_RATE = 1e-4    # CFG模型学习率
    NUM_WORKERS = 0          # CFG模型数据加载进程数
    EPOCHS = 100             # CFG模型训练轮数
    WEIGHT_DECAY = 1e-8      # CFG模型L2正则化
    
    # 包含batch_size和learning_rate的文件路径
    BEST_MODEL_PATH = BaseConfig.MODEL_DIR / f"cfg_gcn_model_bs{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}" / f"best_cfg_model_bs{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY }.pt"
    CHECKPOINT_DIR = BaseConfig.MODEL_DIR / f"cfg_gcn_model_bs{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}" / "checkpoints"
    TRAIN_LOG = BaseConfig.LOG_DIR / f"cfg_gcn_model_bs{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}" / f"training_bs{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}.log"
    
    # 池化层配置
    POOLING_TYPE = 'attention'
    NUM_ATTENTION_HEADS = 1  # CFG使用单头注意力
    



class FCGConfig(BaseConfig):
    """FCG模型专用配置"""
    INPUT_DIM = 300
    EMBEDDING_DIM = 128
    
    # FCG模型固定训练参数
    BATCH_SIZE = 32          # FCG模型固定批大小（相对CFG较小，因为FCG图更复杂）
    LEARNING_RATE = 0.0001   # FCG模型学习率（相对CFG较小，更稳定训练）
    NUM_WORKERS = 0          # FCG模型数据加载进程数
    EPOCHS = 100             # FCG模型训练轮数
    WEIGHT_DECAY = 1e-4      # FCG模型L2正则化
    
    # 包含batch_size和learning_rate的文件路径
    BEST_MODEL_PATH = BaseConfig.MODEL_DIR / f"fcg_gcn_model_bs{BATCH_SIZE}_lr{LEARNING_RATE}" / f"best_fcg_model_bs{BATCH_SIZE}_lr{LEARNING_RATE}.pt"
    CHECKPOINT_DIR = BaseConfig.MODEL_DIR / f"fcg_gcn_model_bs{BATCH_SIZE}_lr{LEARNING_RATE}" / "checkpoints"
    TRAIN_LOG = BaseConfig.LOG_DIR / f"fcg_gcn_model_bs{BATCH_SIZE}_lr{LEARNING_RATE}" / f"training_bs{BATCH_SIZE}_lr{LEARNING_RATE}.log"
    
    # 池化层配置
    POOLING_TYPE = 'attention'
    NUM_ATTENTION_HEADS = 4  # FCG使用多头注意力，更好地捕获函数调用关系
    
