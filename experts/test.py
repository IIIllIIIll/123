import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import random
from datetime import datetime

# -----------------------------
# 0. 固定随机种子
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# -----------------------------
# 1. MoE 核心组件 (与V4/V5相同)
# -----------------------------

class ExpertNetwork(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=64):
        super(ExpertNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GatingNetwork(nn.Module):
    def __init__(self, input_dim=128, num_experts=30):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
    def forward(self, x):
        return F.softmax(self.gate(x), dim=1)


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=64, num_experts=30, k=30):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([ExpertNetwork(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gating = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x):
        gates = self.gating(x)
        _, indices = torch.topk(gates, self.k, dim=1)
        final_output = torch.zeros(x.size(0), self.experts[0](x).size(1), device=x.device)
        for i in range(x.size(0)):
            sample_output = 0
            expert_indices = indices[i]
            expert_weights = gates[i, expert_indices]
            expert_weights = expert_weights / expert_weights.sum()
            for j, expert_idx in enumerate(expert_indices):
                expert_out = self.experts[expert_idx](x[i:i+1])
                sample_output += expert_out * expert_weights[j]
            final_output[i] = sample_output
        return final_output

# -----------------------------
# 2. 数据集 (与之前相同)
# -----------------------------
class TriModalVectorDataset(Dataset):
    def __init__(self, feature_paths, labels_df):
        self.labels_df = labels_df.reset_index(drop=True)
        self.id_list = self.labels_df['sample_id'].tolist()
        self.labels = torch.tensor(self.labels_df['label'].tolist(), dtype=torch.long)
        self.zero_vector = torch.zeros(128, dtype=torch.float)

        print(f"为 {len(self.id_list)} 个样本加载特征...")
        self.binary_map = self.load_features_map(feature_paths['binary'], file_type='h5')
        self.cfg_map = self.load_features_map(feature_paths['cfg'], file_type='h5')
        self.fcg_map = self.load_features_map(feature_paths['fcg'], file_type='h5')
        print("特征加载器初始化完成。")

    def load_features_map(self, path, file_type='h5'):
        feature_map = {}
        try:
            if file_type == 'h5':
                print(f"Loading H5: {path}")
                with h5py.File(path, 'r') as f:
                    features = f['features'][:]
                    # 优先尝试读取 sample_id，如果不存在则尝试 filenames
                    if 'sample_id' in f:
                        sample_ids_bytes = f['sample_id'][:]
                        for sid_b, vec in zip(sample_ids_bytes, features):
                            sid = sid_b.decode('utf-8')
                            feature_map[sid] = torch.tensor(vec, dtype=torch.float)
                    elif 'filenames' in f:
                        filenames_bytes = f['filenames'][:]
                        for fname_b, vec in zip(filenames_bytes, features):
                            fname = fname_b.decode('utf-8')
                            feature_map[fname] = torch.tensor(vec, dtype=torch.float)
                    else:
                        print(f"警告: H5文件 {path} 中既没有 'sample_id' 也没有 'filenames' 字段")
            print(f"Loaded {len(feature_map)} vectors from {path}")
        except FileNotFoundError:
            print(f"警告: 特征文件未找到: {path}. 该模态将全部使用0向量。")
        except Exception as e:
            print(f"错误: 加载 {path} 时出错: {e}. 该模态将全部使用0向量。")
        return feature_map

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        sample_id = self.id_list[idx]
        label = self.labels[idx]
        vec_binary = self.binary_map.get(sample_id, self.zero_vector)
        vec_cfg = self.cfg_map.get(sample_id, self.zero_vector)
        vec_fcg = self.fcg_map.get(sample_id, self.zero_vector)
        return (vec_binary, vec_cfg, vec_fcg, label)

# -----------------------------
# 3. 【新】简单拼接 + MLP 分类器 
# (已替换 MoETransformerClassifier)
# -----------------------------
class SimpleMLPFusionModel(nn.Module):
    def __init__(self, moe_input_dim=128, moe_hidden_dim=64, moe_output_dim=64, 
                 num_experts=30, k=30, num_classes=2, dropout=0.1):
        """
        使用简单拼接 (Concatenation) 和 MLP 头的融合模型。
        
        参数:
        moe_input_dim (int): MoE 专家网络的输入维度 (例如: 128)
        moe_hidden_dim (int): MoE 专家网络的隐藏层维度 (例如: 64)
        moe_output_dim (int): MoE 专家网络的输出维度 (例如: 64)
        num_experts (int): 专家总数
        k (int): 每次激活的专家数
        num_classes (int): 最终分类数 (例如: 2)
        dropout (float): 用于分类头的 dropout 比率
        """
        super().__init__()
        
        # 1. 保留共享的MoE (与 transformer_training.py 相同)
        self.shared_moe = MixtureOfExperts(moe_input_dim, moe_hidden_dim, moe_output_dim, num_experts, k)
        
        # 2. 定义新的MLP分类头
        # 输入维度 = 3个模态 * MoE输出维度 (例如: 3 * 64 = 192)
        fusion_input_dim = moe_output_dim * 3 
        
        self.classifier_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128), # 第一个隐藏层
            nn.BatchNorm1d(128),              # 添加批量归一化
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),               # 第二个隐藏层
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)        # 输出层
        )
        
    def freeze_moe_parameters(self, pretrained_model_path):
        """
        加载预训练的 MoE 权重并冻结它们。
        (与 transformer_training.py 相同)
        """
        try:
            print(f"正在从 {pretrained_model_path} 加载MoE权重...")
            # 加载预训练模型的状态字典
            pretrained_dict = torch.load(pretrained_model_path)
            
            # 过滤出 shared_moe 的权重
            # (注意: V4模型的权重可能没有 'shared_moe.' 前缀, 
            #  如果V4模型就是MoE, 我们可能需要调整键名)
            
            # 尝试自动检测前缀
            moe_dict = {}
            if any(k.startswith('shared_moe.') for k in pretrained_dict.keys()):
                # 如果V4模型保存时有 'shared_moe.' 前缀 (例如它也是一个融合模型)
                moe_dict = {k.replace('shared_moe.', ''): v for k, v in pretrained_dict.items() if k.startswith('shared_moe.')}
            else:
                # 假设V4模型本身就是MoE (例如 TriModalExpertModel)
                # 我们需要知道V4模型中MoE模块的名字, 假设V4的MoE也叫 'shared_moe'
                # 如果V4的MoE模块不叫'shared_moe', 这将失败
                moe_dict = {k.replace('shared_moe.', ''): v for k, v in pretrained_dict.items() if k.startswith('shared_moe.')}
                # 如果V4模型就是纯MoE, 没有'shared_moe.'前缀, 我们需要不同的加载逻辑
                # **一个更鲁棒的假设：V4的 .pt 文件就是 MoE 的 state_dict**
                if not moe_dict:
                    print("未检测到 'shared_moe.' 前缀, 假设 .pt 文件是 MoE state_dict ...")
                    moe_dict = pretrained_dict 

            if not moe_dict:
                 print(f"警告：在 {pretrained_model_path} 中未找到 'shared_moe.' 键。")
                 # 尝试加载 (这可能因为键不匹配而失败)
                 self.shared_moe.load_state_dict(pretrained_dict, strict=False)
            else:
                 self.shared_moe.load_state_dict(moe_dict)

            print("MoE权重加载成功。")

        except Exception as e:
            print(f"警告：加载预训练MoE权重失败: {e}。将使用随机初始化的MoE参数（不推荐）。")
        
        # 冻结参数
        # for param in self.shared_moe.parameters():
        #     param.requires_grad = False
        # print("shared_moe 模块的参数已冻结。")

    def forward(self, vec_binary, vec_cfg, vec_fcg):
        batch_size = vec_binary.size(0)
        
        # 1. 冻结的MoE提取特征
        # (在 .eval() 模式下, no_grad() 是自动的, 但为了清晰, 
        #  在训练时我们也用 no_grad() 来确保冻结)
        with torch.no_grad():
            binary_out = self.shared_moe(vec_binary) # [B, 64]
            cfg_out = self.shared_moe(vec_cfg)       # [B, 64]
            fcg_out = self.shared_moe(vec_fcg)       # [B, 64]
        
        # 2. 拼接特征 (Concatenate)
        # [B, 64], [B, 64], [B, 64] -> [B, 192]
        fused_vector = torch.cat([binary_out, cfg_out, fcg_out], dim=1)
        
        # 3. 通过MLP分类头
        logits = self.classifier_head(fused_vector)
        return logits

# -----------------------------
# 4. 评估函数 (与之前相同)
# -----------------------------
def evaluate_model(model, data_loader, device, desc="Eval"):
    """
    在给定的 data_loader 上评估模型
    返回 (accuracy, precision, recall, f1)
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for vec_binary, vec_cfg, vec_fcg, labels in tqdm(data_loader, desc=desc):
            vec_binary = vec_binary.to(device)
            vec_cfg = vec_cfg.to(device)
            vec_fcg = vec_cfg.to(device)
            labels = labels.to(device)
            
            outputs = model(vec_binary, vec_cfg, vec_fcg)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r1 = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, pre, r1, f1

# -----------------------------
# 5. 训练和评估 (主要逻辑与之前相同)
# -----------------------------
if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)
    
      # --- 配置参数 ---
    BATCH_SIZE = 32
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.001  # L2正则化参数
    DROPOUT_RATE = 0.3   # Dropout比率
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10  # 早停耐心值
    
    # (!! 注意: MoE的输出维度是D_MODEL)
    D_MODEL = 64         # MoE 输出维度
    NUM_EXPERTS = 30
    K_EXPERTS = 6
    
    # (!! 注意: Transformer特定参数已不再需要)
    # NHEAD = 4
    # NUM_LAYERS = 2
    # DIM_FEEDFORWARD = 256
    
    # 生成模型配置标识 (使用新模型名称)
    model_config = f"simple_mlp_fusion_bs{BATCH_SIZE}_lr{LEARNING_RATE}_wd{WEIGHT_DECAY}_dr{DROPOUT_RATE}_experts{NUM_EXPERTS}_k{K_EXPERTS}_dm{D_MODEL}_patience{EARLY_STOPPING_PATIENCE}"
    
    # --- 配置路径 ---
    DATA_ROOT = "/mnt/lbq/output/"
    FEATURE_PATHS = {
        "binary": os.path.join(DATA_ROOT, "feature", "convnext_extracted_features_128d.h5"),
        "cfg": os.path.join(DATA_ROOT, "feature", "cfg_extracted_features_all.h5"),
        "fcg": os.path.join(DATA_ROOT, "feature", "fcg_extracted_features_all.h5")
    }
    LABELS_PATH = os.path.join(DATA_ROOT, "labels.csv")
    # PRETRAINED_MOE_PATH = "/mnt/lbq/output/model/tri_modal_expert_bs32_lr0.0001/best_tri_modal_expert_bs32_lr0.0001.pt" # V4 训练好的模型
    PRETRAINED_MOE_PATH = "/mnt/lbq/output/model/tri_model_expert_bs32_lr5e-07_experts30_k6/best_tri_model_expert_bs32_lr5e-07_experts30_k6.pt"
    
    # 定义模型保存路径
    MODEL_DIR = os.path.join(DATA_ROOT, "model", model_config)
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, f"best_{model_config}.pth")
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, f"final_{model_config}.pth")
    
    # 定义日志文件路径
    LOG_DIR = os.path.join(DATA_ROOT, "log", model_config)
    LOG_FILE_PATH = os.path.join(LOG_DIR, f"{model_config}_training.log")
    SUMMARY_LOG_PATH = os.path.join(LOG_DIR, f"{model_config}_summary.txt")

    # 确保目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 打印配置信息
    print(f"\n=== Simple MLP Fusion 训练配置 ===")
    print(f"模型配置: {model_config}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"权重衰减: {WEIGHT_DECAY}")
    print(f"Dropout比率: {DROPOUT_RATE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"早停耐心值: {EARLY_STOPPING_PATIENCE}")
    print(f"MoE输出维度: {D_MODEL}")
    print(f"专家数量: {NUM_EXPERTS}")
    print(f"激活专家数: {K_EXPERTS}")
    print(f"模型保存目录: {MODEL_DIR}")
    print(f"日志保存目录: {LOG_DIR}")
    print("=" * 50)


    # --- 数据加载  ---
    try:
        all_labels_df = pd.read_csv(LABELS_PATH)
        
        train_df = all_labels_df[all_labels_df['split'] == 'train'].copy()
        val_df = all_labels_df[all_labels_df['split'] == 'val'].copy()
        test_df = all_labels_df[all_labels_df['split'] == 'test'].copy()

        print(f"训练集: {len(train_df)} 样本")
        print(f"验证集: {len(val_df)} 样本")
        print(f"测试集: {len(test_df)} 样本")

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            print("错误: 'train', 'val', 或 'test' 划分中至少有一个为空。请检查 labels.csv。")
            exit(1)

        print("\n--- 初始化训练集 ---")
        train_dataset = TriModalVectorDataset(FEATURE_PATHS, train_df)
        print("\n--- 初始化验证集 ---")
        val_dataset = TriModalVectorDataset(FEATURE_PATHS, val_df)
        print("\n--- 初始化测试集 ---")
        test_dataset = TriModalVectorDataset(FEATURE_PATHS, test_df)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
    except Exception as e:
        print(f"数据加载时发生错误: {e}")
        exit(1)
        
    # --- 模型、损失函数、优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # (!! 注意: 实例化新模型)
    model = SimpleMLPFusionModel(
        moe_input_dim=128,           # 固定的输入维度
        moe_hidden_dim=64,           # MoE隐藏维度
        moe_output_dim=D_MODEL,      # MoE输出维度 (64)
        num_experts=NUM_EXPERTS,
        k=K_EXPERTS,
        num_classes=2,
        dropout=DROPOUT_RATE
    ).to(device)

    # (!! 注意: 冻结MoE参数的调用保持不变)
    model.freeze_moe_parameters(PRETRAINED_MOE_PATH)
    
    criterion = nn.CrossEntropyLoss()
    
    # (!! 注意: 优化器只会优化未冻结的参数，即MLP头的参数)
    optimizer = torch.optim.Adam(
        # filter(lambda p: p.requires_grad, model.parameters()), 
        model.parameters(),
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    # --- 训练和评估循环 (与之前相同) ---
    log_file = open(LOG_FILE_PATH, "w")
    log_file.write("Epoch,Train_ACC,Train_PRE,Train_R1,Train_F1,Val_ACC,Val_PRE,Val_R1,Val_F1,Test_ACC,Test_PRE,Test_R1,Test_F1\n")

    best_val_f1 = 0.0 
    patience_counter = 0
    early_stopped = False 

    print(f"\n--- 开始训练 (Simple-MLP-Fusion: {model_config}) ---")
    for epoch in range(NUM_EPOCHS):
        # 1. 训练阶段
        model.train()
        train_preds = []
        train_labels = []
        for vec_binary, vec_cfg, vec_fcg, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Train"):
            vec_binary, vec_cfg, vec_fcg, labels = vec_binary.to(device), vec_cfg.to(device), vec_cfg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(vec_binary, vec_cfg, vec_fcg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_pre = precision_score(train_labels, train_preds, average='macro', zero_division=0)
        train_r1 = recall_score(train_labels, train_preds, average='macro', zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
        print(f"Epoch {epoch+1} - Train Metrics - ACC: {train_acc:.4f}, F1: {train_f1:.4f}")

        # 2. 验证阶段
        val_acc, val_pre, val_r1, val_f1 = evaluate_model(model, val_loader, device, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Validate")
        print(f"Epoch {epoch+1} - Val Metrics   - ACC: {val_acc:.4f}, F1: {val_f1:.4f}")

        # 3. 测试阶段 (在每个 epoch 报告，用于观察)
        test_acc, test_pre, test_r1, test_f1 = evaluate_model(model, test_loader, device, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Test")
        print(f"Epoch {epoch+1} - Test Metrics  - ACC: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        # 4. 检查点和早停
        if val_f1 > best_val_f1:
            print(f"  !! 新的最佳验证 F1: {val_f1:.4f} (优于 {best_val_f1:.4f})。正在保存模型到 {BEST_MODEL_PATH}...")
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience_counter += 1
            print(f"  验证F1未改善 ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  !! 早停触发：验证F1连续{EARLY_STOPPING_PATIENCE}个epoch未改善，停止训练")
                early_stopped = True
        
        # 5. 记录日志
        log_line = (f"{epoch+1},{train_acc:.4f},{train_pre:.4f},{train_r1:.4f},{train_f1:.4f},"
                    f"{val_acc:.4f},{val_pre:.4f},{val_r1:.4f},{val_f1:.4f},"
                    f"{test_acc:.4f},{test_pre:.4f},{test_r1:.4f},{test_f1:.4f}\n")
        log_file.write(log_line)
        log_file.flush()
        
        # 6. 早停检查
        if early_stopped:
            break

    # 保存最终模型
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"最终模型已保存到: {FINAL_MODEL_PATH}")
    
    log_file.close()
    if early_stopped:
        print(f"\n--- 训练因早停而终止。日志已保存到 {LOG_FILE_PATH} ---")
    else:
        print(f"\n--- 训练完成。日志已保存到 {LOG_FILE_PATH} ---")

    # -----------------------------
    # 6. 评估 (与之前相同)
    # -----------------------------
    print(f"--- 正在加载在验证集上表现最佳的模型 ({BEST_MODEL_PATH}) ... ---")
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        print("最佳模型加载成功。")
    except Exception as e:
        print(f"错误: 加载最佳模型失败: {e}。将使用最后一个epoch的模型进行评估。")

    print("--- 正在测试集上运行最终的公正评估 ---")
    
    final_acc, final_pre, final_r1, final_f1 = evaluate_model(model, test_loader, device, desc="Final Test")
    
    print("\n--- 最终模型评估结果 (基于最佳验证模型) ---")
    print(f"  - 最终测试集 Accuracy: {final_acc:.4f}")
    print(f"  - 最终测试集 Precision: {final_pre:.4f}")
    print(f"  - 最终测试集 Recall: {final_r1:.4f}")
    print(f"  - 最终测试集 F1-Score:  {final_f1:.4f}")
    print("-------------------------------------------------")
    
    # 将最终结果附加到日志文件末尾
    with open(LOG_FILE_PATH, "a") as f:
        f.write("\n--- Final Results (using best model on validation set) ---\n")
        f.write(f"Test_ACC,Test_PRE,Test_R1,Test_F1\n")
        f.write(f"{final_acc:.4f},{final_pre:.4f},{final_r1:.4f},{final_f1:.4f}\n")
    
    # 生成训练总结报告
    with open(SUMMARY_LOG_PATH, "w") as f:
        f.write(f"=== Simple MLP Fusion 训练总结报告 ===\n")
        f.write(f"模型配置: {model_config}\n")
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"批次大小: {BATCH_SIZE}\n")
        f.write(f"学习率: {LEARNING_RATE}\n")
        f.write(f"权重衰减: {WEIGHT_DECAY}\n")
        f.write(f"Dropout比率: {DROPOUT_RATE}\n")
        f.write(f"训练轮数: {NUM_EPOCHS}\n")
        f.write(f"早停耐心值: {EARLY_STOPPING_PATIENCE}\n")
        f.write(f"是否早停: {'是' if early_stopped else '否'}\n")
        f.write(f"MoE输出维度: {D_MODEL}\n")
        f.write(f"专家数量: {NUM_EXPERTS}\n")
        f.write(f"激活专家数: {K_EXPERTS}\n")
        f.write(f"\n=== 最终测试结果 ===\n")
        f.write(f"测试集准确率: {final_acc:.4f}\n")
        f.write(f"测试集精确率: {final_pre:.4f}\n")
        f.write(f"测试集召回率: {final_r1:.4f}\n")
        f.write(f"测试集F1分数: {final_f1:.4f}\n")
        f.write(f"\n=== 文件路径 ===\n")
        f.write(f"最佳模型: {BEST_MODEL_PATH}\n")
        f.write(f"最终模型: {FINAL_MODEL_PATH}\n")
        f.write(f"训练日志: {LOG_FILE_PATH}\n")
        f.write(f"总结报告: {SUMMARY_LOG_PATH}\n")
    
    print(f"训练总结报告已保存到: {SUMMARY_LOG_PATH}")