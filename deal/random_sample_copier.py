#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
随机筛选样本用于fasttext微调
基于标签的样本复制工具
根据CSV标签文件选择训练集样本，保持良性和恶意样本平衡
支持.NET文件比例控制

PE文件处理工具集
"""

import os
import sys
import random
import shutil
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict


def load_labels_csv(csv_path: str) -> pd.DataFrame:
    """
    加载标签CSV文件
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        DataFrame包含样本信息
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功加载CSV文件: {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"❌ 加载CSV文件失败: {e}")
        return pd.DataFrame()


def filter_train_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    筛选训练集样本
    
    Args:
        df: 包含所有样本的DataFrame
        
    Returns:
        仅包含训练集样本的DataFrame
    """
    train_df = df[df['split'] == 'train'].copy()
    print(f"📊 训练集样本数量: {len(train_df)}")
    
    # 统计良性和恶意样本数量
    benign_count = len(train_df[train_df['label'] == 0])
    malware_count = len(train_df[train_df['label'] == 1])
    
    print(f"   - 良性样本: {benign_count}")
    print(f"   - 恶意样本: {malware_count}")
    
    return train_df


def select_balanced_samples(train_df: pd.DataFrame, sample_count: int = 3000, dotnet_ratio: float = 0.2) -> pd.DataFrame:
    """
    选择恶意和良性样本（2:1比例），并控制.NET文件比例
    
    Args:
        train_df: 训练集DataFrame
        sample_count: 总样本数量
        dotnet_ratio: .NET文件占总样本的比例
        
    Returns:
        按2:1比例选择的样本DataFrame（恶意:良性）
    """
    # 分离良性和恶意样本
    benign_samples = train_df[train_df['label'] == 0]
    malware_samples = train_df[train_df['label'] == 1]
    
    # 计算每类样本数量（恶意:良性 = 2:1）
    malware_count = (sample_count * 2) // 3  # 2/3 恶意样本
    benign_count = sample_count - malware_count  # 1/3 良性样本
    
    # 计算.NET文件数量
    dotnet_count = int(sample_count * dotnet_ratio)
    dotnet_malware_count = (dotnet_count * 2) // 3  # 2/3 .NET恶意样本
    dotnet_benign_count = dotnet_count - dotnet_malware_count  # 1/3 .NET良性样本
    
    # 检查是否有足够的样本
    available_benign = len(benign_samples)
    available_malware = len(malware_samples)
    
    # 检查.NET样本数量
    available_benign_dotnet = len(benign_samples[benign_samples['is_dotnet_binary'] == 1])
    available_malware_dotnet = len(malware_samples[malware_samples['is_dotnet_binary'] == 1])
    
    actual_benign = min(benign_count, available_benign)
    actual_malware = min(malware_count, available_malware)
    
    actual_benign_dotnet = min(dotnet_benign_count, available_benign_dotnet)
    actual_malware_dotnet = min(dotnet_malware_count, available_malware_dotnet)
    
    print(f"🎯 目标样本分布 (恶意:良性 = 2:1):")
    print(f"   - 恶意样本: {actual_malware} (可用: {available_malware})")
    print(f"     └─ .NET恶意: {actual_malware_dotnet} (可用: {available_malware_dotnet})")
    print(f"   - 良性样本: {actual_benign} (可用: {available_benign})")
    print(f"     └─ .NET良性: {actual_benign_dotnet} (可用: {available_benign_dotnet})")
    
    # 选择.NET样本
    selected_benign_dotnet = benign_samples[benign_samples['is_dotnet_binary'] == 1].sample(
        n=actual_benign_dotnet, random_state=42) if actual_benign_dotnet > 0 else pd.DataFrame()
    selected_malware_dotnet = malware_samples[malware_samples['is_dotnet_binary'] == 1].sample(
        n=actual_malware_dotnet, random_state=42) if actual_malware_dotnet > 0 else pd.DataFrame()
    
    # 选择非.NET样本
    remaining_benign = actual_benign - actual_benign_dotnet
    remaining_malware = actual_malware - actual_malware_dotnet
    
    # 从非.NET样本中选择
    benign_non_dotnet = benign_samples[benign_samples['is_dotnet_binary'] == 0]
    malware_non_dotnet = malware_samples[malware_samples['is_dotnet_binary'] == 0]
    
    selected_benign_non_dotnet = benign_non_dotnet.sample(
        n=min(remaining_benign, len(benign_non_dotnet)), random_state=42) if remaining_benign > 0 else pd.DataFrame()
    selected_malware_non_dotnet = malware_non_dotnet.sample(
        n=min(remaining_malware, len(malware_non_dotnet)), random_state=42) if remaining_malware > 0 else pd.DataFrame()
    
    # 合并所有选择的样本
    selected_samples = pd.concat([
        selected_benign_dotnet, selected_benign_non_dotnet,
        selected_malware_dotnet, selected_malware_non_dotnet
    ], ignore_index=True)
    
    print(f"✅ 实际选择样本数量: {len(selected_samples)}")
    print(f"   - .NET文件: {len(selected_samples[selected_samples['is_dotnet_binary'] == 1])}")
    print(f"   - 非.NET文件: {len(selected_samples[selected_samples['is_dotnet_binary'] == 0])}")
    
    return selected_samples


def get_all_files(directory: str) -> List[str]:
    """
    获取目录中的所有文件
    
    Args:
        directory: 目录路径
        
    Returns:
        文件路径列表
    """
    try:
        files = []
        for file_path in Path(directory).iterdir():
            if file_path.is_file():
                files.append(str(file_path))
        return files
    except Exception as e:
        print(f"❌ 读取目录失败: {e}")
        return []


def validate_directories(source_dir: str, target_dir: str) -> bool:
    """
    验证源目录和目标目录
    
    Args:
        source_dir: 源目录路径
        target_dir: 目标目录路径
        
    Returns:
        验证是否通过
    """
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"❌ 源目录不存在: {source_dir}")
        return False
    
    if not os.path.isdir(source_dir):
        print(f"❌ 源路径不是目录: {source_dir}")
        return False
    
    # 创建目标目录（如果不存在）
    try:
        os.makedirs(target_dir, exist_ok=True)
        print(f"✅ 目标目录已准备: {target_dir}")
    except Exception as e:
        print(f"❌ 创建目标目录失败: {e}")
        return False
    
    return True


def check_existing_files(target_dir: str) -> int:
    """
    检查目标目录中已存在的文件数量
    
    Args:
        target_dir: 目标目录路径
        
    Returns:
        已存在的文件数量
    """
    try:
        existing_files = [f for f in os.listdir(target_dir) 
                         if os.path.isfile(os.path.join(target_dir, f))]
        return len(existing_files)
    except Exception:
        return 0


def copy_selected_samples(selected_df: pd.DataFrame, benign_source_dir: str, 
                         malware_source_dir: str, target_dir: str, 
                         file_types: List[str] = None, overwrite: bool = False) -> Tuple[int, int, List[str]]:
    """
    根据选择的样本从源目录复制多种类型的JSON文件
    
    Args:
        selected_df: 选择的样本DataFrame
        benign_source_dir: 良性样本源目录路径
        malware_source_dir: 恶意样本源目录路径
        target_dir: 目标目录路径
        file_types: 要复制的文件类型列表，如['_fcg.json', '_ida_analysis.json']
        overwrite: 是否覆盖已存在的文件
        
    Returns:
        (成功复制数量, 跳过数量, 错误列表)
    """
    if file_types is None:
        file_types = ['_fcg.json']  # 默认只复制FCG文件
    
    success_count = 0
    skip_count = 0
    errors = []
    
    total_samples = len(selected_df)
    total_files = total_samples * len(file_types)
    
    print(f"📁 开始复制 {total_samples} 个样本的 {len(file_types)} 种文件类型...")
    print(f"📋 文件类型: {', '.join(file_types)}")
    print(f"📊 预计总文件数: {total_files}")
    
    processed_files = 0
    
    for i, (_, row) in enumerate(selected_df.iterrows(), 1):
        try:
            sample_id = row['sample_id']
            label = row['label']
            is_dotnet = row['is_dotnet_binary']
            
            # 根据标签确定源目录
            if label == 0:  # 良性样本
                label_name = "良性"
                source_dir = benign_source_dir
            else:  # 恶意样本
                label_name = "恶意"
                source_dir = malware_source_dir
            
            if not os.path.exists(source_dir):
                error_msg = f"源目录不存在: {source_dir}"
                errors.append(error_msg)
                processed_files += len(file_types)  # 跳过所有文件类型
                continue
            
            # 复制每种文件类型
            for file_type in file_types:
                processed_files += 1
                
                # 构建源文件和目标文件路径
                source_file = os.path.join(source_dir, f"{sample_id}{file_type}")
                target_file = os.path.join(target_dir, f"{sample_id}{file_type}")
                
                # 检查源文件是否存在
                if not os.path.exists(source_file):
                    error_msg = f"源文件不存在: {sample_id}{file_type} ({label_name}{'/.NET' if is_dotnet else ''})"
                    errors.append(error_msg)
                    if len(errors) <= 10:
                        print(f"❌ {error_msg}")
                    continue
                
                # 检查目标文件是否已存在
                if os.path.exists(target_file) and not overwrite:
                    skip_count += 1
                    if skip_count <= 10:
                        print(f"⏭️  跳过已存在文件: {os.path.basename(target_file)} ({label_name}{'/.NET' if is_dotnet else ''})")
                    continue
                
                # 复制文件
                shutil.copy2(source_file, target_file)
                success_count += 1
                
                # 显示进度
                if processed_files % 50 == 0 or processed_files == total_files:
                    progress = (processed_files / total_files) * 100
                    print(f"📈 进度: {processed_files}/{total_files} ({progress:.1f}%) - 成功: {success_count}, 跳过: {skip_count}")
                
        except Exception as e:
            error_msg = f"处理样本失败 {sample_id}: {str(e)}"
            errors.append(error_msg)
            if len(errors) <= 10:
                print(f"❌ {error_msg}")
    
    return success_count, skip_count, errors


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="基于标签的样本复制工具 - 根据CSV标签文件选择训练集样本，保持良性和恶意样本平衡",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python random_sample_copier.py --csv "path/to/labels.csv" --benign "path/to/benign" --malware "path/to/malware" --target "path/to/target"
  python random_sample_copier.py -c "labels.csv" -b "benign_dir" -m "malware_dir" -t "target_dir" --count 6000
        """
    )
    
    parser.add_argument(
        "-c", "--csv",
        default=r"d:\Test\AAAAAAAAAAAAAAA\demooooooo\output\labels_with_dotnet.csv",
        help="标签CSV文件路径 (默认: labels_with_dotnet.csv)"
    )
    
    parser.add_argument(
        "--benign-source",
        default=r"d:\Test\AAAAAAAAAAAAAAA\demooooooo\dataset\benign_ida_analysis_cfg",
        help="良性样本源目录路径 (默认: benign_ida_analysis_cfg目录)"
    )
    
    parser.add_argument(
        "--malware-source",
        default=r"d:\Test\AAAAAAAAAAAAAAA\demooooooo\dataset\malware_ida_analysis_cfg",
        help="恶意样本源目录路径 (默认: malware_ida_analysis_cfg目录)"
    )
    
    parser.add_argument(
        "-t", "--target", 
        default=r"d:\Test\AAAAAAAAAAAAAAA\demooooooo\dataset\fasttext_data_controlFG",
        help="目标目录路径 (默认: fasttext_data_controlFG目录)"
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=600,
        help="要复制的总样本数量 (默认: 600，恶意:良性=2:1)"
    )
    
    parser.add_argument(
        "--file-types",
        nargs='+',
        default=['_ida_analysis.json'],
        help="要复制的文件类型列表 (默认: ['_ida_analysis.json']，可选: '_cfg.json'等)"
    )
    
    parser.add_argument(
        "--dotnet-ratio",
        type=float,
        default=0.2,
        help=".NET文件占总样本的比例 (默认: 0.2)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的文件"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于可重现的随机选择 (默认: 42)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 基于标签的多文件类型样本复制工具")
    print("=" * 60)
    print(f"📋 标签文件: {args.csv}")
    print(f"📂 良性样本源目录: {args.benign_source}")
    print(f"📂 恶意样本源目录: {args.malware_source}")
    print(f"📁 目标目录: {args.target}")
    print(f"🔢 总样本数量: {args.count} (恶意: {(args.count*2)//3}, 良性: {args.count - (args.count*2)//3})")
    print(f"📋 文件类型: {', '.join(args.file_types)}")
    print(f"🔄 覆盖模式: {'是' if args.overwrite else '否'}")
    print(f"🌱 随机种子: {args.seed}")
    print("-" * 60)
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 验证CSV文件
    if not os.path.exists(args.csv):
        print(f"❌ 标签CSV文件不存在: {args.csv}")
        sys.exit(1)
    
    # 验证源目录
    if not os.path.exists(args.benign_source):
        print(f"❌ 良性样本源目录不存在: {args.benign_source}")
        sys.exit(1)
        
    if not os.path.exists(args.malware_source):
        print(f"❌ 恶意样本源目录不存在: {args.malware_source}")
        sys.exit(1)
    
    # 创建目标目录
    try:
        os.makedirs(args.target, exist_ok=True)
        print(f"✅ 目标目录已准备: {args.target}")
    except Exception as e:
        print(f"❌ 创建目标目录失败: {e}")
        sys.exit(1)
    
    # 加载CSV文件
    print("\n🔍 正在加载标签文件...")
    df = load_labels_csv(args.csv)
    if df.empty:
        sys.exit(1)
    
    # 筛选训练集样本
    print("\n📊 正在筛选训练集样本...")
    train_df = filter_train_samples(df)
    if train_df.empty:
        print("❌ 没有找到训练集样本")
        sys.exit(1)
    
    # 选择平衡的样本
    print("\n🎯 正在选择平衡样本...")
    selected_df = select_balanced_samples(train_df, args.count)
    if selected_df.empty:
        print("❌ 没有选择到样本")
        sys.exit(1)
    
    # 检查目标目录中已存在的文件
    existing_count = check_existing_files(args.target)
    if existing_count > 0:
        print(f"\n📋 目标目录中已有 {existing_count} 个文件")
        if not args.overwrite:
            print("💡 使用 --overwrite 参数可覆盖已存在的文件")
    
    # 用户确认
    try:
        print(f"\n📋 即将复制的样本:")
        benign_selected = len(selected_df[selected_df['label'] == 0])
        malware_selected = len(selected_df[selected_df['label'] == 1])
        total_files_to_copy = len(selected_df) * len(args.file_types)
        print(f"   - 恶意样本: {malware_selected}")
        print(f"   - 良性样本: {benign_selected}")
        print(f"   - 总样本数: {len(selected_df)}")
        print(f"   - 文件类型数: {len(args.file_types)}")
        print(f"   - 预计复制文件总数: {total_files_to_copy}")
        
        confirm = input(f"\n❓ 确认要复制这些样本吗？(y/N): ").strip().lower()
        if confirm not in ['y', 'yes', '是']:
            print("❌ 操作已取消")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n❌ 操作已取消")
        sys.exit(0)
    
    print("\n" + "=" * 60)
    
    # 执行复制
    success_count, skip_count, errors = copy_selected_samples(
        selected_df, args.benign_source, args.malware_source, args.target, 
        args.file_types, args.overwrite
    )
    
    # 显示结果
    print("\n" + "=" * 60)
    print("📊 复制完成统计")
    print("=" * 60)
    print(f"✅ 成功复制: {success_count} 个文件")
    print(f"⏭️  跳过文件: {skip_count} 个文件")
    print(f"❌ 复制失败: {len(errors)} 个文件")
    
    if errors:
        print(f"\n❌ 错误详情 (显示前10个):")
        for error in errors[:10]:
            print(f"   • {error}")
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors) - 10} 个错误")
    
    # 验证结果
    final_count = check_existing_files(args.target)
    print(f"\n📁 目标目录最终文件数量: {final_count}")
    
    if success_count > 0:
        print("🎉 基于标签的样本复制完成！")
    else:
        print("⚠️  没有文件被复制")
        sys.exit(1)


if __name__ == "__main__":
    main()