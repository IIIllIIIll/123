import torch
import torch.nn as nn
import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import sys

# --- 动态导入模型 ---
# 假设训练脚本与此脚本在同一目录或Python路径中
try:
    from tri_modal_expert_training import TriModalExpertModel
    from transformer_training import MoETransformerClassifier
except ImportError:
    print("错误：无法导入模型类。")
    print("请确保 tri_modal_expert_training.py 和 transformer_training.py 在您的Python路径中。")
    sys.exit(1)

# --- 路径定义 ---
base_dir = Path('/mnt/lbq')
labels_path = base_dir / 'output' / 'labels_pro.csv'
# labels_path = base_dir / 'output' / 'labels.csv'
paper_tri_modal_model_path = base_dir / 'output' / 'model' / 'paper_tri_model_expert_bs32_lr5e-07_experts50_k15' / 'best_tri_model_expert_bs32_lr5e-07_experts50_k15.pt'
moe_transformer_model_path = base_dir / 'output' / 'model' / 'moe_transformer_bs32_lr1e-06_wd0.01_dr0.2_experts50_k15_dm64_nh4_nl1_patience10' / 'best_moe_transformer_bs32_lr1e-06_wd0.01_dr0.2_experts50_k15_dm64_nh4_nl1_patience10.pth'
cfg_features_path = base_dir / 'output' / 'feature' / 'cfg_extracted_features_all.h5'
convnext_features_path = base_dir / 'output' / 'feature' / 'convnext_extracted_features_128d.h5'
fcg_features_path = base_dir / 'output' / 'feature' / 'fcg_extracted_features_all.h5'

def load_labels(labels_path):
    """加载并过滤测试集标签"""
    df = pd.read_csv(labels_path)
    test_df = df[df['split'] == 'test']
    print(f"从 {labels_path} 加载了 {len(test_df)} 个测试样本标签。")
    return test_df['sample_id'].tolist(), test_df['label'].astype(int).tolist()

# --- 关键修正：使用训练脚本中的H5加载逻辑 ---
def load_feature_map_from_h5(path):
    """
    将H5文件（包含 'features' 和 'sample_id' 数据集）
    加载到内存中的字典 (map) 中。
    """
    feature_map = {}
    try:
        print(f"正在加载 H5 特征图: {path}")
        with h5py.File(path, 'r') as f:
            if 'features' in f and ('sample_id' in f or 'filenames' in f):
                features = f['features'][:]
                
                # 确定ID键
                id_key = 'sample_id' if 'sample_id' in f else 'filenames'
                sample_ids_bytes = f[id_key][:]
                
                for sid_b, vec in zip(sample_ids_bytes, features):
                    # 解码ID（以防是bytes）
                    sid = sid_b.decode('utf-8') if isinstance(sid_b, bytes) else str(sid_b)
                    feature_map[sid] = np.array(vec)
            else:
                # 备用逻辑：如果H5是旧的“字典”格式
                print(f"警告: 在 {path} 中未找到 'features'/'sample_id'。")
                print("将尝试直接将H5的根键作为样本ID...")
                for key in f.keys():
                    feature_map[key] = np.array(f[key])
        
        print(f"从 {path} 加载了 {len(feature_map)} 个向量。")
            
    except FileNotFoundError:
        print(f"错误: 特征文件未找到: {path}。")
    except Exception as e:
        print(f"错误: 加载 {path} 时出错: {e}")
            
    return feature_map

def evaluate_model(model, cfg_feats, conv_feats, fcg_feats, true_labels):
    """评估模型性能"""
    model.eval()
    with torch.no_grad():
        # 将Numpy数组转换为Tensors
        cfg_tensor = torch.tensor(cfg_feats).float()
        conv_tensor = torch.tensor(conv_feats).float()
        fcg_tensor = torch.tensor(fcg_feats).float()

        # 将数据移动到模型所在的设备
        device = next(model.parameters()).device
        cfg_tensor = cfg_tensor.to(device)
        conv_tensor = conv_tensor.to(device)
        fcg_tensor = fcg_tensor.to(device)

        # 假定模型 forward 方法的输入顺序是 cfg, conv, fcg
        # !! 如果您的 MoETransformerClassifier 期望不同的输入，请在此处调整 !!
        inputs = [cfg_tensor, conv_tensor, fcg_tensor]
        
        # 捕获 TriModalExpertModel 和 MoETransformerClassifier 不同的返回值
        if isinstance(model, TriModalExpertModel):
             # TriModalExpertModel 返回 (output, dist_loss, eq_loss)
             outputs, _, _ = model(cfg_tensor, conv_tensor, fcg_tensor)
        elif isinstance(model, MoETransformerClassifier):
             # MoETransformerClassifier 只返回 logits
             outputs = model(cfg_tensor, conv_tensor, fcg_tensor)
        else:
             print(f"未知的模型类型: {type(model)}")
             return 0,0,0,0,None

        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds)
    rec = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    cm = confusion_matrix(true_labels, preds)
    return acc, prec, rec, f1, cm

def main():
    sample_ids, true_labels_list = load_labels(labels_path)
    
    print("\n--- 正在将特征加载到内存 ---")
    cfg_map = load_feature_map_from_h5(cfg_features_path)
    conv_map = load_feature_map_from_h5(convnext_features_path)
    fcg_map = load_feature_map_from_h5(fcg_features_path)

    # --- 修正后的数据过滤逻辑（使用 .get() 从 map 中获取） ---
    original_count = len(sample_ids)
    valid_cfg_feats = []
    valid_conv_feats = []
    valid_fcg_feats = []
    valid_labels = []
    dropped_samples_log = []

    for i in range(original_count):
        sid = sample_ids[i]
        
        # 从字典中安全地获取特征
        cfg_vec = cfg_map.get(sid)
        conv_vec = conv_map.get(sid)
        fcg_vec = fcg_map.get(sid)

        # 仅当一个样本的所有三个特征都存在时，才将其保留
        if cfg_vec is not None and conv_vec is not None and fcg_vec is not None:
            valid_cfg_feats.append(cfg_vec)
            valid_conv_feats.append(conv_vec)
            valid_fcg_feats.append(fcg_vec)
            valid_labels.append(true_labels_list[i])
        else:
            # 记录哪些特征缺失了
            missing = []
            if cfg_vec is None: missing.append("CFG")
            if conv_vec is None: missing.append("ConvNext")
            if fcg_vec is None: missing.append("FCG")
            dropped_samples_log.append(f"{sid} (Missing: {', '.join(missing)})")

    # 过滤后，将列表转换为 Numpy 数组
    cfg_feats = np.array(valid_cfg_feats)
    conv_feats = np.array(valid_conv_feats)
    fcg_feats = np.array(valid_fcg_feats)
    true_labels = valid_labels
    
    print(f"\n--- 数据过滤报告 ---")
    print(f"原始测试集大小: {original_count}")
    print(f"具有完整三模态特征的样本数: {len(true_labels)}")
    print(f"因特征缺失被丢弃的样本数: {original_count - len(true_labels)}")
    
    if dropped_samples_log:
        print("\n被丢弃的样本示例:")
        for log_entry in dropped_samples_log[:5]: # 最多显示5个
            print(f"  {log_entry}")
    print(f"--------------------------------\n")

    if len(true_labels) == 0:
        print("错误: 没有有效的样本可供评估。请检查所有H5特征文件。")
        return
    # --- 过滤逻辑结束 ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        # 加载 tri_modal_expert 模型
        # 参数必须与训练时完全一致: experts50_k15
        tri_model = TriModalExpertModel(input_dim=128, moe_hidden_dim=64, moe_output_dim=64, num_experts=50, k=15, fusion_dim=128, num_classes=2)
        tri_model.load_state_dict(torch.load(paper_tri_modal_model_path, map_location=device))
        tri_model.to(device)
        tri_model.eval()
        
        # 加载 moe_transformer 模型
        moe_model = MoETransformerClassifier(moe_input_dim=128, moe_hidden_dim=64, moe_output_dim=64, num_experts=50, k=15, d_model=64, nhead=4, num_layers=1, dim_feedforward=256, num_classes=2, dropout=0.2)
        moe_model.load_state_dict(torch.load(moe_transformer_model_path, map_location=device))
        moe_model.to(device)
        moe_model.eval()
        
        print("正在评估 Paper Tri-Modal Expert (在过滤后的数据上):")
        tri_acc, tri_prec, tri_rec, tri_f1, tri_cm = evaluate_model(tri_model, cfg_feats, conv_feats, fcg_feats, true_labels)
        print(f"Accuracy: {tri_acc:.4f}, Precision: {tri_prec:.4f}, Recall: {tri_rec:.4f}, F1: {tri_f1:.4f}")
        print("Confusion Matrix:\n", tri_cm)
        
        print("\n正在评估 MoE Transformer (在过滤后的数据上):")
        # !! 注意 !!: 确保您的 MoETransformerClassifier 模型的 forward() 方法
        # 接受 (cfg, conv, fcg) 三个输入。如果它只接受一个融合后的输入，
        # 您需要修改这里的 evaluate_model 调用。
        moe_acc, moe_prec, moe_rec, moe_f1, moe_cm = evaluate_model(moe_model, cfg_feats, conv_feats, fcg_feats, true_labels)
        print(f"Accuracy: {moe_acc:.4f}, Precision: {moe_prec:.4f}, Recall: {moe_rec:.4f}, F1: {moe_f1:.4f}")
        print("Confusion Matrix:\n", moe_cm)
        
        # 比较
        print("\n--- 比较 ---")
        print(f"Paper Tri-Modal Expert F1: {tri_f1:.4f} vs MoE Transformer F1: {moe_f1:.4f}")
        print(f"Paper Tri-Modal Expert Acc: {tri_acc:.4f} vs MoE Transformer Acc: {moe_acc:.4f}")

    except FileNotFoundError as e:
        print(f"错误: 无法加载模型文件: {e}")
    except RuntimeError as e:
        print(f"模型加载或评估时发生运行时错误: {e}")
        print("!! 请检查您模型定义中的参数（如input_dim, num_experts 等）是否与您加载的 .pth 文件完全匹配 !!")
    except Exception as e:
        print(f"发生意外错误: {e}")

if __name__ == "__main__":
    main()