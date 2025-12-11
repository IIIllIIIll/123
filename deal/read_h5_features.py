#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5特征文件读取器
用于读取convnext_extracted_features_128d.h5文件中的特征数据
"""

import h5py
import numpy as np
import os

def read_h5_file_info(h5_file_path):
    """
    读取H5文件的基本信息和结构
    """
    print(f"正在读取H5文件: {h5_file_path}")
    print("=" * 60)
    
    if not os.path.exists(h5_file_path):
        print(f"错误: 文件不存在 - {h5_file_path}")
        return None
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            print("H5文件结构:")
            print("-" * 40)
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"数据集: {name}")
                    print(f"  形状: {obj.shape}")
                    print(f"  数据类型: {obj.dtype}")
                    print(f"  大小: {obj.size}")
                elif isinstance(obj, h5py.Group):
                    print(f"组: {name}")
            
            f.visititems(print_structure)
            
            # 获取所有键
            keys = list(f.keys())
            print(f"\n可用的键: {keys}")
            
            return f, keys
            
    except Exception as e:
        print(f"读取H5文件时出错: {e}")
        return None

def read_first_n_features(h5_file_path, n=10):
    """
    读取H5文件中的前n条特征数据
    """
    print(f"\n正在读取前{n}条特征数据...")
    print("=" * 60)
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # 检查可用的数据集
            if 'features' in f:
                features = f['features']
                print(f"特征数据集形状: {features.shape}")
                print(f"特征数据类型: {features.dtype}")
                
                # 读取前n条特征
                if len(features) >= n:
                    first_n_features = features[:n]
                    print(f"\n前{n}条特征数据:")
                    print("-" * 40)
                    
                    for i, feature in enumerate(first_n_features):
                        print(f"特征 {i+1}:")
                        print(f"  形状: {feature.shape}")
                        print(f"  数据类型: {feature.dtype}")
                        print(f"  最小值: {np.min(feature):.6f}")
                        print(f"  最大值: {np.max(feature):.6f}")
                        print(f"  均值: {np.mean(feature):.6f}")
                        print(f"  标准差: {np.std(feature):.6f}")
                        
                        # 显示完整的128个特征值
                        print(f"  完整的128个特征值:")
                        # 每行显示10个值，便于阅读
                        for j in range(0, len(feature), 10):
                            end_idx = min(j + 10, len(feature))
                            values_str = ", ".join([f"{val:.6f}" for val in feature[j:end_idx]])
                            print(f"    [{j:3d}-{end_idx-1:3d}]: {values_str}")
                        print()
                    
                    return first_n_features
                else:
                    print(f"警告: 文件中只有 {len(features)} 条特征，少于请求的 {n} 条")
                    return features[:]
            
            # 检查其他可能的键
            elif 'sample_id' in f and 'labels' in f:
                print("发现sample_id和labels数据集")
                
                sample_ids = f['sample_id']
                labels = f['labels']
                
                print(f"样本ID数据集形状: {sample_ids.shape}")
                print(f"标签数据集形状: {labels.shape}")
                
                # 读取前n条
                if len(sample_ids) >= n:
                    first_n_ids = sample_ids[:n]
                    first_n_labels = labels[:n]
                    
                    print(f"\n前{n}条样本信息:")
                    print("-" * 40)
                    
                    for i in range(n):
                        sample_id = first_n_ids[i]
                        label = first_n_labels[i]
                        
                        # 处理字节字符串
                        if isinstance(sample_id, bytes):
                            sample_id = sample_id.decode('utf-8')
                        
                        print(f"样本 {i+1}:")
                        print(f"  ID: {sample_id}")
                        print(f"  标签: {label}")
                        print()
                    
                    return first_n_ids, first_n_labels
                else:
                    print(f"警告: 文件中只有 {len(sample_ids)} 条记录，少于请求的 {n} 条")
                    return sample_ids[:], labels[:]
            
            else:
                print("未找到预期的数据集。可用的键:")
                for key in f.keys():
                    dataset = f[key]
                    print(f"  {key}: 形状={dataset.shape}, 类型={dataset.dtype}")
                
                # 尝试读取第一个数据集的前n条
                first_key = list(f.keys())[0]
                first_dataset = f[first_key]
                
                if len(first_dataset) >= n:
                    return first_dataset[:n]
                else:
                    return first_dataset[:]
                    
    except Exception as e:
        print(f"读取特征数据时出错: {e}")
        return None

def main():
    """
    主函数
    """
    h5_file_path = r"d:\Test\AAAAAAAAAAAAAAA\demooooooo\output\feature\convnext_extracted_features_128d.h5"
    
    # 检查文件信息
    file_info = read_h5_file_info(h5_file_path)
    
    if file_info is not None:
        # 读取前10条特征
        features = read_first_n_features(h5_file_path, 10)
        
        if features is not None:
            print("\n特征数据读取完成!")
        else:
            print("\n特征数据读取失败!")
    else:
        print("无法读取H5文件!")

if __name__ == "__main__":
    main()