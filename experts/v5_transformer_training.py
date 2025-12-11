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

# 不使用验证集，直接在测试集上评估

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
# 1. MoE 核心组件 (与V4相同, 用于加载V4模型权重)
# -----------------------------

#
class ExpertNetwork(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=64):
        super(ExpertNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#
class GatingNetwork(nn.Module):
    def __init__(self, input_dim=128, num_experts=30):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        
    def forward(self, x):
        return F.softmax(self.gate(x), dim=1)

#
class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=64, num_experts=30, k=30):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.k = k
        
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, output_dim) 
            for _ in range(num_experts)
        ])
        
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
# 2. 数据集 (与V4相同, 加载原始128维向量)
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
                    filenames_bytes = f['filenames'][:]
                    for fname_b, vec in zip(filenames_bytes, features):
                        fname = fname_b.decode('utf-8')
                        feature_map[fname] = torch.tensor(vec, dtype=torch.float)
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
# 3. 新模型：MoE-Transformer 分类器
# -----------------------------

class MoETransformerClassifier(nn.Module):
    """
    两阶段模型：
    1. 冻结的MoE：将 (B, 128) * 3 转换为 (B, 3, 64) 的序列
    2. 可训练的Transformer：分析 (B, 3, 64) 序列并进行分类
    """
    def __init__(self, 
                 # MoE 参数 (必须与V4匹配)
                 moe_input_dim=128, moe_hidden_dim=64, moe_output_dim=64, 
                 num_experts=30, k=30,
                 # Transformer 参数
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=256,
                 num_classes=2, dropout=0.1):
        
        super().__init__()
        
        self.moe_output_dim = moe_output_dim
        
        # 阶段1：MoE 特征转换器 (将被冻结)
        self.shared_moe = MixtureOfExperts(
            input_dim=moe_input_dim,
            hidden_dim=moe_hidden_dim,
            output_dim=moe_output_dim,
            num_experts=num_experts,
            k=k
        )
        
        # -----------------------------------
        # 阶段2：Transformer 序列分类器 (可训练)
        # -----------------------------------
        
        # [CLS] Token: 一个可学习的向量，用于汇总序列信息
        # (1, 1, d_model) -> 对应 (Seq_len, Batch, Dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 位置编码: (Seq_len + 1, Batch, Dim)
        # 我们有 3 个模态 + 1 个 [CLS] token = 4
        # (4, 1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(4, 1, d_model))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False # PyTorch 默认 (Seq, Batch, Dim)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 分类器头部 (只接收 [CLS] token 的输出)
        self.classifier_head = nn.Linear(d_model, num_classes)
        
        # 初始化 [CLS] 和位置编码
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

    def freeze_moe_parameters(self, pretrained_model_path):
        """
        加载预训练的MoE权重并冻结它们
        """
        try:
            print(f"正在从 {pretrained_model_path} 加载MoE权重...")
            pretrained_dict = torch.load(pretrained_model_path)
            
            # 仅提取 'shared_moe.' 开头的键
            moe_dict = {k.replace('shared_moe.', ''): v 
                        for k, v in pretrained_dict.items() 
                        if k.startswith('shared_moe.')}
            
            self.shared_moe.load_state_dict(moe_dict)
            print("MoE权重加载成功。")

        except Exception as e:
            print(f"警告：加载预训练MoE权重失败: {e}")
            print("将使用随机初始化的MoE参数（不推荐）。")

        # 冻结所有MoE参数
        for param in self.shared_moe.parameters():
            param.requires_grad = False
        print("shared_moe 模块的参数已冻结。")


    def forward(self, vec_binary, vec_cfg, vec_fcg):
        batch_size = vec_binary.size(0)
        
        # -----------------------------------
        # 阶段1: 特征转换 (使用冻结的MoE)
        # -----------------------------------
        # requires_grad=False 会自动传播, 但使用 no_grad() 更明确
        with torch.no_grad():
            binary_out = self.shared_moe(vec_binary) # (B, 64)
            cfg_out = self.shared_moe(vec_cfg)       # (B, 64)
            fcg_out = self.shared_moe(vec_fcg)       # (B, 64)
        
        # (B, 3, 64) -> (3, B, 64) 以匹配 PyTorch Transformer
        x = torch.stack([binary_out, cfg_out, fcg_out], dim=0)

        # -----------------------------------
        # 阶段2: Transformer 序列分析
        # -----------------------------------
        
        # 1. 准备 [CLS] token 并扩展到批次大小
        # (1, 1, 64) -> (1, B, 64)
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)
        
        # 2. 拼接 [CLS] token 和 序列
        # (1, B, 64) + (3, B, 64) -> (4, B, 64)
        x = torch.cat((cls_tokens, x), dim=0)
        
        # 3. 添加位置编码
        # (4, B, 64) + (4, 1, 64) -> (4, B, 64)
        x = x + self.pos_embedding
        
        # 4. 通过 Transformer
        transformer_out = self.transformer_encoder(x) # (4, B, 64)
        
        # 5. 提取 [CLS] token 的输出 (序列中的第0个)
        # (4, B, 64) -> (B, 64)
        cls_output = transformer_out[0]
        
        # 6. 分类
        logits = self.classifier_head(cls_output) # (B, 2)
        
        return logits

# -----------------------------
# 4. 训练和评估
# -----------------------------
if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)
    
    # --- 配置路径 ---
    DATA_ROOT = "/mnt/data1_l20_raid5disk/lbq_dataset/output/"
    
    FEATURE_PATHS = {
        "binary": os.path.join(DATA_ROOT, "feature", "all_binary_features_128d.h5"),
        "cfg": os.path.join(DATA_ROOT, "feature", "all_cfg_features_128d.h5"),
        "fcg": os.path.join(DATA_ROOT, "feature", "all_fcg_features_128d.h5")
    }
    
    LABELS_PATH = os.path.join(DATA_ROOT, "labels.csv")
    PRETRAINED_MOE_PATH = "./tri_modal_expert_v4_model_30e.pth" # V4 训练好的模型

    # --- 数据加载 (使用V4的CSV划分) ---
    try:
        all_labels_df = pd.read_csv(LABELS_PATH)
        
        train_df = all_labels_df[all_labels_df['split'] == 'train'].copy()
        val_df = all_labels_df[all_labels_df['split'] == 'val'].copy()
        test_df = all_labels_df[all_labels_df['split'] == 'test'].copy()

        print(f"训练集: {len(train_df)} 样本")
        print(f"验证集: {len(val_df)} 样本")
        print(f"测试集: {len(test_df)} 样本")

        if len(test_df) == 0 and len(val_df) > 0:
            print("警告: 'test' 划分样本为 0。将使用 'val' 划分作为测试集。")
            test_df = val_df
        elif len(test_df) == 0 and len(val_df) == 0:
             print("错误: 'test' 和 'val' 划分均为空！无法进行评估。")
             exit(1)

        print("\n--- 初始化训练集 ---")
        train_dataset = TriModalVectorDataset(FEATURE_PATHS, train_df)
        print("\n--- 初始化测试集 ---")
        test_dataset = TriModalVectorDataset(FEATURE_PATHS, test_df)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
        
    except Exception as e:
        print(f"数据加载时发生错误: {e}")
        exit(1)
        
    # --- 模型、损失函数、优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # 1. 创建新模型
    model = MoETransformerClassifier(
        d_model=64,           # 必须匹配 MoE 的 moe_output_dim
        moe_output_dim=64,    # 必须匹配 MoE
        num_experts=30,       # 必须匹配 MoE
        k=30,                 # 必须匹配 MoE
        nhead=4,              # Transformer 头数
        num_layers=2,         # Transformer 层数
        dim_feedforward=256,  # Transformer 前馈网络维度
        num_classes=2
    ).to(device)

    # 2. 加载预训练权重并冻结 MoE
    model.freeze_moe_parameters(PRETRAINED_MOE_PATH)

    criterion = nn.CrossEntropyLoss()
    
    # 3. !! 关键：只为可训练的参数创建优化器 !!
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=1e-4
    )
    
    print("\n--- 将要训练的参数 ---")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    print("------------------------")

    # --- 训练和评估循环 ---
    log_file = open("./moe_transformer_v5_training.log", "w")
    log_file.write("Epoch,Train_ACC,Train_PRE,Train_R1,Train_F1,Test_ACC,Test_PRE,Test_R1,Test_F1\n")

    print("\n--- 开始训练 (仅训练Transformer部分) ---")
    for epoch in range(100):
        model.train()
        all_preds = []
        all_labels = []

        # (训练循环与V4相同)
        for vec_binary, vec_cfg, vec_fcg, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            vec_binary = vec_binary.to(device)
            vec_cfg = vec_cfg.to(device)
            vec_fcg = vec_fcg.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(vec_binary, vec_cfg, vec_fcg) # (B, 128), (B, 128), (B, 128)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(all_labels, all_preds)
        train_pre = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        train_r1 = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1} - Train Metrics - ACC: {train_acc:.4f}, PRE: {train_pre:.4f}, R1: {train_r1:.4f}, F1: {train_f1:.4f}")

        # 测试阶段
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for vec_binary, vec_cfg, vec_fcg, labels in tqdm(test_loader, desc=f"Epoch {epoch+1} Test "):
                vec_binary = vec_binary.to(device)
                vec_cfg = vec_cfg.to(device)
                vec_fcg = vec_fcg.to(device)
                labels = labels.to(device)
                
                outputs = model(vec_binary, vec_cfg, vec_fcg)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        test_pre = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        test_r1 = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch+1} - Test Metrics - ACC: {test_acc:.4f}, PRE: {test_pre:.4f}, R1: {test_r1:.4f}, F1: {test_f1:.4f}")
        
        log_file.write(f"{epoch+1},{train_acc:.4f},{train_pre:.4f},{train_r1:.4f},{train_f1:.4f},{test_acc:.4f},{test_pre:.4f},{test_r1:.4f},{test_f1:.4f}\n")
        log_file.flush()

    log_file.close()
    
    # 保存整个模型（包括冻结的MoE和训练好的Transformer）
    torch.save(model.state_dict(), "./moe_transformer_v5_model.pth")
    print("Transformer 模型训练完成并已保存。")