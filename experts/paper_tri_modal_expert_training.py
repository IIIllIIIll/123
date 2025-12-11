import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd 
import h5py 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import random

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
# 1. MoE 核心组件 (已按论文修改)
# -----------------------------

def distinctiveness_loss(expert_outputs):
    """
    计算专家之间的多样性损失 (Distinctiveness Loss)
    !! 已更新：严格按照论文 Eq (11) 和 (12) 实现 [cite: 187, 198, 199, 200]
    Eq (11): R_H^2 = -E[k(e, e_hat)]
    
    参数:
    expert_outputs: list of tensors, 每个专家的平均输出
    """
    if len(expert_outputs) < 2:
        # 如果活跃专家少于2个，无法计算成对损失
        return torch.tensor(0.0, device=expert_outputs[0].device if expert_outputs else 'cpu')
    
    loss = 0
    num_experts = len(expert_outputs)
    num_pairs = 0
    
    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            expert_i = expert_outputs[i]
            expert_j = expert_outputs[j]
            
            # k(e, e_hat) - 高斯核相似性
            similarity = torch.exp(-torch.norm(expert_i - expert_j, dim=1) ** 2 / 2)
            
            # 根据 Eq (11)，我们取相似度的负均值 
            loss -= similarity.mean()
            num_pairs += 1
    
    # 根据 Eq (12) [cite: 198, 199, 200]，我们计算所有对的平均值
    if num_pairs > 0:
        loss = loss / num_pairs
    
    return loss

def equilibrium_loss(expert_weights, expert_indices):
    """
    计算专家平衡损失 (Equilibrium Loss)
    !! 已更新：严格按照论文 Eq (7), (9), (10) 实现 
    Eq (10): l_e = (1/M) * sum(rho_i * rho_hat_i)
    
    参数:
    expert_weights: Tensor [V*N, M], Gating 输出的概率 (pi)
    expert_indices: Tensor [V*N, K], Top-K 选中的专家索引
    """
    # V*N 是总样本数 (跨视图)，M 是专家总数
    total_samples, M = expert_weights.size()
    
    if total_samples == 0:
        return torch.tensor(0.0, device=expert_weights.device)

    # 1. 计算 rho_hat_i (Eq 9) [cite: 164]
    # rho_hat_i = sum_v sum_j pi_i(x_j^v)
    # expert_weights 已经是所有 v 和 j 的 pi，所以我们按专家维度求和
    rho_hat_i = expert_weights.sum(dim=0) # 结果维度 [M]

    # 2. 计算 rho_i (Eq 7) [cite: 156]
    # rho_i = sum_v sum_j M_ji^v
    # M_ji^v 是 one-hot mask (基于 expert_indices) [cite: 158, 160]
    rho_i = torch.zeros(M, device=expert_weights.device)
    
    # 将 [V*N, K] 的索引转换为 [V*N, M] 的 one-hot 编码，并按 K 维度求和
    # (一个样本可能选择多个专家，所以 sum(dim=1) 而不是 squeeze)
    one_hot_mask = F.one_hot(expert_indices, num_classes=M).sum(dim=1) # 维度 [V*N, M]
    
    # 再按样本维度求和
    rho_i = one_hot_mask.sum(dim=0) # 结果维度 [M]

    # 3. 计算 Eq (10) 
    # l_e = (1/M) * sum(rho_i * rho_hat_i)
    loss = (1.0 / M) * (rho_i * rho_hat_i).sum()
    
    return loss


class ExpertNetwork(nn.Module):
    """ 专家网络：一个简单的全连接网络 """
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=64):
        super(ExpertNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GatingNetwork(nn.Module):
    """ 门控网络：决定每个专家的权重（由外部传入控制） """
    def __init__(self, input_dim=128, num_experts=None):
        super(GatingNetwork, self).__init__()
        # 若未显式传入，则使用合理的默认值（与模型保持一致）
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, self.num_experts)
        
    def forward(self, x):
        return F.softmax(self.gate(x), dim=1)


class MixtureOfExperts(nn.Module):
    """ 混合专家层：结合门控和多个专家（专家数量与Top-K可配置） """
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=64, num_experts=None, k=None):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.k = k
        
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, output_dim) 
            for _ in range(self.num_experts)
        ])
        
        self.gating = GatingNetwork(input_dim, self.num_experts)
        
    def forward(self, x):
        gates = self.gating(x)
        _, indices = torch.topk(gates, self.k, dim=1)
        final_output = torch.zeros(x.size(0), self.experts[0](x).size(1), device=x.device)
        
        # !! 修改：收集每个专家的原始输出，用于全局多样性损失
        expert_outputs_per_expert = [[] for _ in range(self.num_experts)]
        
        for i in range(x.size(0)):
            sample_output = 0
            expert_indices = indices[i]
            expert_weights = gates[i, expert_indices]
            expert_weights = expert_weights / expert_weights.sum()
            
            for j, expert_idx in enumerate(expert_indices):
                expert_out = self.experts[expert_idx](x[i:i+1])
                sample_output += expert_out * expert_weights[j]
                
                # !! 修改：收集原始输出
                expert_outputs_per_expert[expert_idx].append(expert_out)
                
            final_output[i] = sample_output
        
        # !! 修改：不再在此处计算损失
        # 而是返回计算损失所需的所有原始材料
        
        return final_output, expert_outputs_per_expert, gates, indices

# -----------------------------
# 2. 数据集 (未修改)
# -----------------------------

class TriModalVectorDataset(Dataset):
    """
    用于加载三个模态（Binary, CFG, FCG）的128维特征向量的数据集。
    以一个 DataFrame 作为基准。
    如果 allow_missing=True，则缺失特征用0向量填充；否则过滤掉不完整的样本。
    """
    def __init__(self, feature_paths, labels_df, allow_missing=False):
        # 1. 设置基准 (Master List)
        self.labels_df = labels_df.reset_index(drop=True)
        self.id_list = self.labels_df['sample_id'].tolist() # 使用 'sample_id'
        self.labels = torch.tensor(self.labels_df['label'].tolist(), dtype=torch.long)
        
        # 2. 定义0向量 (用于填充缺失)
        self.zero_vector = torch.zeros(128, dtype=torch.float)
        self.allow_missing = allow_missing

        # 3. 加载所有特征到内存中的字典 (Map)
        print(f"为 {len(self.id_list)} 个样本加载特征...")
        self.binary_map = self.load_features_map(feature_paths['binary'], file_type='h5')
        self.cfg_map = self.load_features_map(feature_paths['cfg'], file_type='h5')
        self.fcg_map = self.load_features_map(feature_paths['fcg'], file_type='h5')
        
        if not self.allow_missing:
            # 4. 过滤只保留所有三个模态都存在的样本
            all_complete_ids = set(self.binary_map.keys()) & set(self.cfg_map.keys()) & set(self.fcg_map.keys())
            self.labels_df = self.labels_df[self.labels_df['sample_id'].isin(all_complete_ids)].reset_index(drop=True)
            self.id_list = self.labels_df['sample_id'].tolist()
            self.labels = torch.tensor(self.labels_df['label'].tolist(), dtype=torch.long)
            
            print(f"过滤后完整样本数量: {len(self.id_list)}")
        
        print("特征加载器初始化完成。")

    def load_features_map(self, path, file_type='h5'):
        feature_map = {}
        try:
            if file_type == 'h5':
                print(f"Loading H5: {path}")
                with h5py.File(path, 'r') as f:
                    features = f['features'][:]
                    # 尝试使用 sample_id 而不是 filenames
                    if 'sample_id' in f:
                        sample_ids_bytes = f['sample_id'][:]
                        for sid_b, vec in zip(sample_ids_bytes, features):
                            sid = sid_b.decode('utf-8') if isinstance(sid_b, bytes) else str(sid_b)
                            feature_map[sid] = torch.tensor(vec, dtype=torch.float)
                    elif 'filenames' in f:
                        filenames_bytes = f['filenames'][:]
                        for fname_b, vec in zip(filenames_bytes, features):
                            fname = fname_b.decode('utf-8') if isinstance(fname_b, bytes) else str(fname_b)
                            feature_map[fname] = torch.tensor(vec, dtype=torch.float)
                    else:
                        print(f"警告: H5文件 {path} 中既没有 'sample_id' 也没有 'filenames' 数据集")
                        return feature_map
            
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
        
        if self.allow_missing:
            vec_binary = self.binary_map.get(sample_id, self.zero_vector)
            vec_cfg = self.cfg_map.get(sample_id, self.zero_vector)
            vec_fcg = self.fcg_map.get(sample_id, self.zero_vector)
        else:
            vec_binary = self.binary_map[sample_id]
            vec_cfg = self.cfg_map[sample_id]
            vec_fcg = self.fcg_map[sample_id]
        
        return (vec_binary, vec_cfg, vec_fcg, label)

# -----------------------------
# 5. 融合模型 (!! 已修改 !!)
# -----------------------------

class TriModalExpertModel(nn.Module):
    def __init__(self, input_dim=128, moe_hidden_dim=64, moe_output_dim=64, 
                 num_experts=None, k=None, fusion_dim=128, num_classes=2):
        super(TriModalExpertModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        self.shared_moe = MixtureOfExperts(
            input_dim=input_dim,
            hidden_dim=moe_hidden_dim,
            output_dim=moe_output_dim,
            num_experts=num_experts,
            k=k
        )
        
        self.fusion_input_dim = 3 * moe_output_dim
        self.fusion = nn.Linear(self.fusion_input_dim, fusion_dim)
        
        self.classifier = nn.Linear(fusion_dim, num_classes)
        
    def forward(self, vec_binary, vec_cfg, vec_fcg):
        
        # 1. 分别通过 MoE
        binary_out, binary_expert_outs, binary_gates, binary_indices = self.shared_moe(vec_binary)
        cfg_out, cfg_expert_outs, cfg_gates, cfg_indices = self.shared_moe(vec_cfg)
        fcg_out, fcg_expert_outs, fcg_gates, fcg_indices = self.shared_moe(vec_cfg)
        
        # 2. 融合用于分类
        combined = torch.cat([binary_out, cfg_out, fcg_out], dim=1)
        fused = F.relu(self.fusion(combined))
        output = self.classifier(fused)
        
        # 3. !! 修改：全局计算辅助损失 !!
        
        # --- 3a. Equilibrium Loss (Eq 10) ---
        # 论文的 Eq (7) 和 (9) 是对所有视图 V 和所有样本 N 求和 
        # 因此我们把所有视图的 gates 和 indices 拼接起来，计算一次
        all_gates = torch.cat([binary_gates, cfg_gates, fcg_gates], dim=0)
        all_indices = torch.cat([binary_indices, cfg_indices, fcg_indices], dim=0)
        
        eq_loss = equilibrium_loss(all_gates, all_indices)

        # --- 3b. Distinctiveness Loss (Eq 11 & 12) ---
        # 论文的 Eq (12) 是计算所有专家对 (i, j) 之间的损失 
        # 专家的分布 P_E_i 是基于它处理过的 *所有* 数据 
        
        # 合并所有视图的专家输出
        all_expert_outputs_list = [[] for _ in range(self.num_experts)]
        for i in range(self.num_experts):
            all_expert_outputs_list[i].extend(binary_expert_outs[i])
            all_expert_outputs_list[i].extend(cfg_expert_outs[i])
            all_expert_outputs_list[i].extend(fcg_expert_outs[i])

        # 计算全局平均专家输出
        avg_expert_outputs = []
        for outs in all_expert_outputs_list:
            if outs: # 只有被使用过的专家才参与计算
                stacked = torch.cat(outs, dim=0)
                avg = torch.mean(stacked, dim=0, keepdim=True)
                avg_expert_outputs.append(avg)
        
        dist_loss = distinctiveness_loss(avg_expert_outputs)
        
        return output, dist_loss, eq_loss

# -----------------------------
# 4. 训练和评估 (未修改)
# -----------------------------
if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)
    
    # --- !!! 配置您的路径 !!! ---
    DATA_ROOT = "/mnt/lbq/output/"
    
    FEATURE_PATHS = {
        "binary": "/mnt/lbq/output/feature/convnext_extracted_features_128d.h5",
        "cfg": "/mnt/lbq/output/feature/cfg_extracted_features_all.h5",
        "fcg": "/mnt/lbq/output/feature/fcg_extracted_features_all.h5"
    }
    
    LABELS_PATH = os.path.join(DATA_ROOT, "labels.csv")
    
    # --- 配置输出路径 ---
    MODEL_NAME = "tri_model_expert"
    BATCH_SIZE = 32
    LEARNING_RATE = (1e-7)*5
    # --- 全局专家配置（只需在此修改即可全局生效） ---
    NUM_EXPERTS = 60
    K_EXPERTS = 20

    # 创建输出目录
    LOG_DIR = f"/mnt/lbq/output/log/paper_{MODEL_NAME}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_experts{NUM_EXPERTS}_k{K_EXPERTS}/"
    MODEL_DIR = f"/mnt/lbq/output/model/paper_{MODEL_NAME}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_experts{NUM_EXPERTS}_k{K_EXPERTS}/"
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 输出文件路径
    LOG_FILE_PATH = os.path.join(LOG_DIR, f"training_bs{BATCH_SIZE}_lr{LEARNING_RATE}_experts{NUM_EXPERTS}_k{K_EXPERTS}.log")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, f"best_{MODEL_NAME}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_experts{NUM_EXPERTS}_k{K_EXPERTS}.pt")
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final_bs{BATCH_SIZE}_lr{LEARNING_RATE}_experts{NUM_EXPERTS}_k{K_EXPERTS}.pt")
    
    # --- 数据加载  ---
    try:
        # 1. 读取主标签文件
        all_labels_df = pd.read_csv(LABELS_PATH)
        print(f"总共 {len(all_labels_df)} 个样本（来自 {LABELS_PATH}）。")
        
        # 2. !! 按 'split' 列过滤 DataFrame (不再使用 train_test_split) !!
        train_df = all_labels_df[all_labels_df['split'] == 'train'].copy()
        val_df = all_labels_df[all_labels_df['split'] == 'val'].copy()
        test_df = all_labels_df[all_labels_df['split'] == 'test'].copy()

        # 检查划分
        print(f"训练集: {len(train_df)} 样本 (来自 CSV 'train' 划分)")
        print(f"验证集: {len(val_df)} 样本 (来自 CSV 'val' 划分)")
        print(f"测试集: {len(test_df)} 样本 (来自 CSV 'test' 划分)")

        # 如果 'test' 集为空，则使用 'val' 集作为测试集
        if len(test_df) == 0:
            if len(val_df) > 0:
                print("警告: 'test' 划分样本为 0。将使用 'val' 划分作为测试集。")
                test_df = val_df
            else:
                print("错误: 'test' 和 'val' 划分均为空！无法进行评估。")
                exit(1)
        
        if len(train_df) == 0:
            print("错误: 'train' 划分样本为 0！无法进行训练。")
            exit(1)

        # 3. 创建数据集实例
        print("\n--- 初始化训练集 ---")
        train_dataset = TriModalVectorDataset(FEATURE_PATHS, train_df, allow_missing=False)
        print("\n--- 初始化验证集 ---")
        val_dataset = TriModalVectorDataset(FEATURE_PATHS, val_df, allow_missing=True) if len(val_df) > 0 else None
        print("\n--- 初始化测试集 ---")
        test_dataset = TriModalVectorDataset(FEATURE_PATHS, test_df, allow_missing=True)

        # 4. 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
        
    except FileNotFoundError as e:
        print(f"错误：找不到主标签文件！ {e}")
        print(f"请确保 {LABELS_PATH} 存在。")
        exit(1)
    except Exception as e:
        print(f"数据加载时发生未知错误: {e}")
        print("请检查 FEATURE_PATHS 和 LABELS_PATH 中的文件名和列名是否正确。")
        exit(1)
        
    # --- 模型、损失函数、优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)



    # Loss weights
    lambda_dist = 0.1
    lambda_eq = 0.1

    # 创建模型（下游模块将使用上述配置）
    model = TriModalExpertModel(
        input_dim=128,
        moe_hidden_dim=64,
        moe_output_dim=64,
        num_experts=NUM_EXPERTS,
        k=K_EXPERTS,
        fusion_dim=128,
        num_classes=2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 训练和评估循环 (标准训练/验证/测试流程) ---
    log_file = open(LOG_FILE_PATH, "w")
    if val_loader:
        log_file.write("Epoch,Train_ACC,Train_PRE,Train_R1,Train_F1,Val_ACC,Val_PRE,Val_R1,Val_F1,Test_ACC,Test_PRE,Test_R1,Test_F1\n")
    else:
        log_file.write("Epoch,Train_ACC,Train_PRE,Train_R1,Train_F1,Test_ACC,Test_PRE,Test_R1,Test_F1\n")

    # 早停机制变量
    best_val_acc = 0.0
    best_test_acc = 0.0
    patience = 15
    patience_counter = 0

    print("\n--- 开始训练 ---")
    for epoch in range(300):
        # === 训练阶段 ===
        model.train()
        train_preds = []
        train_labels = []

        for vec_binary, vec_cfg, vec_fcg, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            vec_binary = vec_binary.to(device)
            vec_cfg = vec_cfg.to(device)
            vec_fcg = vec_fcg.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # !! 模型现在返回按论文公式计算的损失
            outputs, dist_loss, eq_loss = model(vec_binary, vec_cfg, vec_fcg)
            
            classification_loss = criterion(outputs, labels)
            
            # !! 训练目标保持不变，但 dist_loss 和 eq_loss 的计算方式已更新
            total_loss = classification_loss + lambda_dist * dist_loss + lambda_eq * eq_loss
            
            total_loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # 计算训练指标
        train_acc = accuracy_score(train_labels, train_preds)
        train_pre = precision_score(train_labels, train_preds, average='macro', zero_division=0)
        train_r1 = recall_score(train_labels, train_preds, average='macro', zero_division=0)
        train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1} - Train Metrics - ACC: {train_acc:.4f}, PRE: {train_pre:.4f}, R1: {train_r1:.4f}, F1: {train_f1:.4f}")

        # === 验证阶段 ===
        val_acc = val_pre = val_r1 = val_f1 = 0.0
        if val_loader:
            model.eval()
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for vec_binary, vec_cfg, vec_fcg, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val  "):
                    vec_binary = vec_binary.to(device)
                    vec_cfg = vec_cfg.to(device)
                    vec_fcg = vec_fcg.to(device)
                    labels = labels.to(device)

                    outputs, _, _ = model(vec_binary, vec_cfg, vec_fcg)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            # 计算验证指标
            val_acc = accuracy_score(val_labels, val_preds)
            val_pre = precision_score(val_labels, val_preds, average='macro', zero_division=0)
            val_r1 = recall_score(val_labels, val_preds, average='macro', zero_division=0)
            val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

            print(f"Epoch {epoch+1} - Val Metrics   - ACC: {val_acc:.4f}, PRE: {val_pre:.4f}, R1: {val_r1:.4f}, F1: {val_f1:.4f}")

            # 早停机制
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳验证模型
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"新的最佳验证准确率: {best_val_acc:.4f} - 模型已保存到 {BEST_MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"验证准确率连续 {patience} 个 epoch 未提升，提前停止训练")
                    break

        # === 测试阶段 ===
        model.eval()
        test_preds = []
        test_labels = []

        with torch.no_grad():
            for vec_binary, vec_cfg, vec_fcg, labels in tqdm(test_loader, desc=f"Epoch {epoch+1} Test "):
                vec_binary = vec_binary.to(device)
                vec_cfg = vec_cfg.to(device)
                vec_fcg = vec_cfg.to(device)
                labels = labels.to(device)

                outputs, _, _ = model(vec_binary, vec_cfg, vec_fcg)
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # 计算测试指标
        test_acc = accuracy_score(test_labels, test_preds)
        test_pre = precision_score(test_labels, test_preds, average='macro', zero_division=0)
        test_r1 = recall_score(test_labels, test_preds, average='macro', zero_division=0)
        test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1} - Test Metrics  - ACC: {test_acc:.4f}, PRE: {test_pre:.4f}, R1: {test_r1:.4f}, F1: {test_f1:.4f}")

        # 记录最佳测试准确率
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # 写入日志
        if val_loader:
            log_file.write(f"{epoch+1},{train_acc:.4f},{train_pre:.4f},{train_r1:.4f},{train_f1:.4f},{val_acc:.4f},{val_pre:.4f},{val_r1:.4f},{val_f1:.4f},{test_acc:.4f},{test_pre:.4f},{test_r1:.4f},{test_f1:.4f}\n")
        else:
            log_file.write(f"{epoch+1},{train_acc:.4f},{train_pre:.4f},{train_r1:.4f},{train_f1:.4f},{test_acc:.4f},{test_pre:.4f},{test_r1:.4f},{test_f1:.4f}\n")
        log_file.flush()

        print("-" * 80)

    log_file.close()
    
    # 保存最终模型
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"\n训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.4f}" if val_loader else "未使用验证集")
    print(f"最佳测试准确率: {best_test_acc:.4f}")
    print(f"最终模型已保存为: {FINAL_MODEL_PATH}")
    if val_loader:
        print(f"最佳验证模型已保存为: {BEST_MODEL_PATH}")
    print(f"训练日志已保存为: {LOG_FILE_PATH}")

    # 最终评估：使用最佳模型在测试集上进行评估（如果有验证集）
    if val_loader:
        print(f"--- 正在加载在验证集上表现最佳的模型 ({BEST_MODEL_PATH}) ... ---")
        try:
            model.load_state_dict(torch.load(BEST_MODEL_PATH))
            print("最佳模型加载成功。")
        except Exception as e:
            print(f"错误: 加载最佳模型失败: {e}。将使用最后一个epoch的模型进行评估。")

        print("--- 正在测试集上运行最终的公正评估 ---")
        
        model.eval()
        final_preds = []
        final_labels = []

        with torch.no_grad():
            for vec_binary, vec_cfg, vec_fcg, labels in tqdm(test_loader, desc="Final Test"):
                vec_binary = vec_binary.to(device)
                vec_cfg = vec_cfg.to(device)
                vec_fcg = vec_fcg.to(device)
                labels = labels.to(device)

                outputs, _, _ = model(vec_binary, vec_cfg, vec_fcg)
                _, preds = torch.max(outputs, 1)
                final_preds.extend(preds.cpu().numpy())
                final_labels.extend(labels.cpu().numpy())

        # 计算最终指标
        final_acc = accuracy_score(final_labels, final_preds)
        final_pre = precision_score(final_labels, final_preds, average='macro', zero_division=0)
        final_r1 = recall_score(final_labels, final_preds, average='macro', zero_division=0)
        final_f1 = f1_score(final_labels, final_preds, average='macro', zero_division=0)

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
    else:
        print("未使用验证集，未进行最佳模型的最终评估。")