#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PE文件到RGB特征图预处理脚本
将PE文件转换为(224,224,3)的RGB特征图并保存为.npy文件
"""

import os
import sys
import glob
import numpy as np
import cv2
import pefile
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import math

# === 常量定义 ===
TARGET_DIM = (224, 224)  # 目标图像尺寸
BASE_DIM = (512, 512)    # 基础图像尺寸
STRUCT_DIM = (64, 64)    # 结构图像尺寸
STANDARD_BYTE_LENGTH = 512 * 512  # 标准字节长度
ENTROPY_BLOCK_SIZE = 8   # 熵计算块大小

def calculate_shannon_entropy(data):
    """计算数据的香农熵"""
    if not data:
        return 0
    
    # 统计字节频率
    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probabilities = byte_counts / len(data)
    
    # 计算熵
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    
    # 归一化到0-255范围
    return int(min(255, entropy * 32))

def scale_value(value, max_val=2**32):
    """将数值缩放到0-255范围"""
    if value == 0:
        return 0
    return int(min(255, (value / max_val) * 255))

def generate_hybrid_feature_map(file_path):
    """
    从PE文件生成混合特征图
    返回: (224, 224, 3) 的numpy数组，或None（如果处理失败）
    """
    pe = None
    try:
        # 加载PE文件
        pe = pefile.PE(file_path, fast_load=True)
        
        # === 阶段 1: 提取原始字节数据 ===
        
        # 1.1 提取精确的四个重要节
        important_sections_data = []
        for section in pe.sections:
            if (section.Name.startswith(b'.text') or 
                section.Name.startswith(b'.rdata') or
                section.Name.startswith(b'.idata') or
                section.Name.startswith(b'.data')):
                important_sections_data.append(section.get_data())
        
        if not important_sections_data:
            with open(file_path, 'rb') as f:
                raw_bytes = f.read(STANDARD_BYTE_LENGTH)
        else:
            raw_bytes = b''.join(important_sections_data)

        # 1.2 标准化长度 (截断或填充)
        if len(raw_bytes) > STANDARD_BYTE_LENGTH:
            raw_bytes = raw_bytes[:STANDARD_BYTE_LENGTH]
        elif len(raw_bytes) < STANDARD_BYTE_LENGTH:
            raw_bytes += b'\x00' * (STANDARD_BYTE_LENGTH - len(raw_bytes))
            
        # 1.3 创建 512x512 基础图
        byte_stream = np.frombuffer(raw_bytes, dtype=np.uint8)
        base_map = byte_stream.reshape(BASE_DIM)

        # === 阶段 2: 生成三个通道 ===

        # --- 通道 R (纹理) ---
        # 使用 INTER_AREA (区域插值) 平滑缩小
        channel_R = cv2.resize(base_map, TARGET_DIM, interpolation=cv2.INTER_AREA)

        # --- 通道 B (熵) ---
        entropy_map = np.zeros(STRUCT_DIM, dtype=np.uint8)
        for r in range(STRUCT_DIM[0]):
            for c in range(STRUCT_DIM[1]):
                r_start, c_start = r * ENTROPY_BLOCK_SIZE, c * ENTROPY_BLOCK_SIZE
                block = base_map[r_start:r_start+ENTROPY_BLOCK_SIZE, 
                                 c_start:c_start+ENTROPY_BLOCK_SIZE]
                entropy_map[r, c] = calculate_shannon_entropy(block.tobytes())
        # 使用 INTER_NEAREST (最近邻) 放大以保持块状
        channel_B = cv2.resize(entropy_map, TARGET_DIM, interpolation=cv2.INTER_NEAREST)

        # --- 通道 G (高密度结构) ---
        struct_map = np.zeros(STRUCT_DIM, dtype=np.uint8) # 64x64 画布
        
        # 组 1: 文件头 (行 0-1)
        struct_map[0, 0] = pe.FILE_HEADER.Machine % 256
        struct_map[0, 1] = pe.FILE_HEADER.Characteristics & 0xFF
        struct_map[0, 2] = (pe.FILE_HEADER.Characteristics >> 8) & 0xFF
        struct_map[1, 0] = pe.OPTIONAL_HEADER.Subsystem
        struct_map[1, 1] = pe.OPTIONAL_HEADER.DllCharacteristics & 0xFF
        struct_map[1, 2] = (pe.OPTIONAL_HEADER.DllCharacteristics >> 8) & 0xFF
        struct_map[1, 3] = 1 if pe.OPTIONAL_HEADER.Magic == 0x10b else 2 # PE32/PE64+
        struct_map[1, 4] = scale_value(pe.FILE_HEADER.SizeOfOptionalHeader)
        
        # 组 2: 关键RVA和大小 (行 2-3)
        size_of_image = pe.OPTIONAL_HEADER.SizeOfImage
        entry_point = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        # struct_map[2, 0] = int((entry_point / size_of_image) * 255) if size_of_image > 0 else 0
        # 计算比率
        ratio = (entry_point / size_of_image) * 255 if size_of_image > 0 else 0
        # 显式地将值钳位在 [0, 255] 范围内
        struct_map[2, 0] = min(max(int(ratio), 0), 255)
        struct_map[2, 1] = scale_value(size_of_image)
        struct_map[2, 2] = scale_value(pe.OPTIONAL_HEADER.SizeOfHeaders)
        # ImageBase 很大, 取 log 后再取模
        struct_map[2, 3] = int(np.log1p(pe.OPTIONAL_HEADER.ImageBase)) % 256
        
        # 入口点位置指纹
        entry_in_text = 0
        entry_in_last = 0
        if pe.sections:
            last_section_idx = len(pe.sections) - 1
            for i, section in enumerate(pe.sections):
                if section.VirtualAddress <= entry_point < (section.VirtualAddress + section.Misc_VirtualSize):
                    if section.Name.startswith(b'.text'):
                        entry_in_text = 255
                    if i == last_section_idx:
                        entry_in_last = 255
                    break
        struct_map[3, 0] = entry_in_text
        struct_map[3, 1] = entry_in_last
        struct_map[3, 2] = 255 if pe.OPTIONAL_HEADER.CheckSum != 0 else 0

        # 组 3: 数据目录 (行 4-5) - 共16个
        if hasattr(pe.OPTIONAL_HEADER, 'DataDirectory'):
            for i, directory in enumerate(pe.OPTIONAL_HEADER.DataDirectory):
                if i >= 16: break
                struct_map[4, i] = scale_value(directory.VirtualAddress)
                struct_map[5, i] = scale_value(directory.Size)

        # 组 4: 导入/导出表摘要 (行 8)
        dll_count, func_count, export_count = 0, 0, 0
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            dll_count = len(pe.DIRECTORY_ENTRY_IMPORT)
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                func_count += len(entry.imports)
        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            export_count = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
        struct_map[8, 0] = scale_value(dll_count)
        struct_map[8, 1] = scale_value(func_count)
        struct_map[8, 2] = scale_value(export_count)

        # 组 5: 节表 (行 16-47) - 最多前16个节
        for i, section in enumerate(pe.sections):
            if i >= 16: break
            base_row = 16 + (i * 2)
            
            # 第1行: 节名称 (8字节) + 虚拟大小 + 原始大小
            name_bytes = section.Name.ljust(8, b'\x00')[:8]
            for j in range(8):
                struct_map[base_row, j] = name_bytes[j]
            struct_map[base_row, 8] = scale_value(section.Misc_VirtualSize)
            struct_map[base_row, 9] = scale_value(section.SizeOfRawData)
            
            # 第2行: 熵 + 4字节的节权限 (Characteristics)
            base_row += 1
            try:
                struct_map[base_row, 0] = calculate_shannon_entropy(section.get_data())
            except Exception:
                struct_map[base_row, 0] = 0 # 空节
            
            flags = section.Characteristics
            struct_map[base_row, 1] = flags & 0xFF
            struct_map[base_row, 2] = (flags >> 8) & 0xFF
            struct_map[base_row, 3] = (flags >> 16) & 0xFF
            struct_map[base_row, 4] = (flags >> 24) & 0xFF
        
        # 使用 INTER_NEAREST (最近邻) 放大以保持块状
        channel_G = cv2.resize(struct_map, TARGET_DIM, interpolation=cv2.INTER_NEAREST)

        # === 阶段 3: 组装最终图像 ===
        final_image = np.stack([channel_R, channel_G, channel_B], axis=-1)
        
        return final_image.astype(np.uint8)

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None
    finally:
        if pe:
            pe.close()

def process_worker(file_path, input_dir, output_dir):
    """
    单个文件处理工作单元。
    """
    try:
        # 生成输出文件名
        relative_path = os.path.relpath(file_path, input_dir)
        safe_filename = relative_path.replace(os.sep, '_') + '.npy'
        output_path = os.path.join(output_dir, safe_filename)
        
        if os.path.exists(output_path):
            return (file_path, "Skipped")
            
        image = generate_hybrid_feature_map(file_path)
        
        if image is not None:
            np.save(output_path, image)
            return (file_path, "Success")
        else:
            return (file_path, "Failed (Invalid PE)")
            
    except Exception as e:
        return (file_path, f"Failed (Error: {e})")

def main():
    """主执行函数"""
    
    # 定义输入输出路径
    base_dir = "/mnt/data1_l20_raid5disk/lbq_dataset"
    
    PATHS = [
        {
            "input": os.path.join(base_dir, "dataset", "benign"),
            "output": os.path.join(base_dir, "dataset", "benign_RGB")
        },
        {
            "input": os.path.join(base_dir, "dataset", "malware"),
            "output": os.path.join(base_dir, "dataset", "malware_RGB")
        }
    ]
    
    # 使用固定的8个线程
    num_cores = 8
    print(f"--- 开始PE文件预处理，使用 {num_cores} 个CPU核心 ---")
    
    total_success = 0
    total_failed = 0
    total_skipped = 0
    
    for path_info in PATHS:
        input_dir = path_info["input"]
        output_dir = path_info["output"]
        
        print(f"\n正在处理: {input_dir} -> {output_dir}")
        
        # 检查输入目录是否存在
        if not os.path.exists(input_dir):
            print(f"警告：输入目录不存在: {input_dir}")
            continue
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print("正在收集文件...")
        file_list = glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
        file_list = [f for f in file_list if os.path.isfile(f)]
        
        if not file_list:
            print("警告：未在输入目录中找到文件。")
            continue
            
        print(f"共找到 {len(file_list)} 个文件。")
        
        # 创建工作函数
        worker_func = partial(process_worker, input_dir=input_dir, output_dir=output_dir)
        
        # 多进程处理
        with Pool(processes=num_cores) as pool:
            results = list(tqdm(pool.imap_unordered(worker_func, file_list), 
                                total=len(file_list), 
                                desc=f"处理 {os.path.basename(input_dir)}"))
        
        # 统计结果
        success_count = sum(1 for _, status in results if status == "Success")
        failed_count = sum(1 for _, status in results if status.startswith("Failed"))
        skipped_count = sum(1 for _, status in results if status == "Skipped")
        
        total_success += success_count
        total_failed += failed_count
        total_skipped += skipped_count
        
        print(f"处理完成: 成功 {success_count}, 失败 {failed_count}, 跳过 {skipped_count}")
    
    print(f"\n--- 预处理全部完成 ---")
    print(f"总计: 成功 {total_success}, 失败 {total_failed}, 跳过 {total_skipped}")

if __name__ == "__main__":
    main()