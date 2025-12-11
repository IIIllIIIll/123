#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.NET二进制检测脚本
用于为CSV文件添加is_dotnet_binary标签

该脚本检查每个样本文件是否只导入了mscoree.dll，以判断是否为.NET二进制文件
"""

import os
import sys
import pandas as pd
import pefile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dotnet_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DotNetDetector:
    """
    .NET二进制文件检测器
    """
    
    def __init__(self, dataset_path: str, output_path: str):
        """
        初始化检测器
        
        Args:
            dataset_path: 数据集路径
            output_path: 输出路径
        """
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.all_data_path = self.dataset_path / "all_data"
        
        # .NET相关的DLL名称
        self.dotnet_dlls = {
            "mscoree.dll",
            "mscoree.dll__corexemain"
        }
        
        logger.info(f"初始化.NET检测器")
        logger.info(f"数据集路径: {self.dataset_path}")
        logger.info(f"输出路径: {self.output_path}")
        logger.info(f"样本文件路径: {self.all_data_path}")
    
    def is_dotnet_binary(self, file_path: Path) -> Tuple[bool, str]:
        """
        检查文件是否为.NET二进制文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Tuple[bool, str]: (是否为.NET文件, 错误信息)
        """
        try:
            # 检查文件是否存在
            if not file_path.exists():
                return False, f"文件不存在: {file_path}"
            
            # 使用pefile解析PE文件
            pe = pefile.PE(str(file_path))
            
            # 获取导入的DLL列表
            imported_dlls = set()
            
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    dll_name = entry.dll.decode('utf-8').lower()
                    imported_dlls.add(dll_name)
            
            pe.close()
            
            # 检查是否只导入了.NET相关的DLL
            if len(imported_dlls) == 1:
                imported_dll = list(imported_dlls)[0]
                if imported_dll in self.dotnet_dlls:
                    logger.debug(f"检测到.NET文件: {file_path.name}, 导入DLL: {imported_dll}")
                    return True, ""
            
            # 如果导入了多个DLL或者不是.NET相关DLL，则不是.NET文件
            logger.debug(f"非.NET文件: {file_path.name}, 导入DLL数量: {len(imported_dlls)}")
            return False, ""
            
        except pefile.PEFormatError as e:
            return False, f"PE格式错误: {str(e)}"
        except Exception as e:
            return False, f"解析错误: {str(e)}"
    
    def find_sample_file(self, sample_id: str) -> Optional[Path]:
        """
        查找样本文件，支持无扩展名和有扩展名的情况
        
        Args:
            sample_id: 样本ID
            
        Returns:
            Optional[Path]: 找到的文件路径，如果未找到返回None
        """
        # 首先尝试直接匹配（无扩展名）
        direct_path = self.all_data_path / sample_id
        if direct_path.exists():
            return direct_path
        
        # 如果直接匹配失败，尝试查找带扩展名的文件
        # 常见的可执行文件扩展名
        common_extensions = [
            '.exe', '.dll', '.bin', '.com', '.scr', '.bat', '.cmd', '.ps1', 
            '.vbs', '.js', '.jar', '.apk', '.elf', '.so', '.dylib', '.msi',
            '.cab', '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'
        ]
        
        for ext in common_extensions:
            file_path = self.all_data_path / f"{sample_id}{ext}"
            if file_path.exists():
                logger.debug(f"找到带扩展名的文件: {sample_id}{ext}")
                return file_path
        
        # 如果还是找不到，使用glob模式匹配
        try:
            matching_files = list(self.all_data_path.glob(f"{sample_id}.*"))
            if matching_files:
                logger.debug(f"通过glob找到文件: {matching_files[0].name}")
                return matching_files[0]
        except Exception as e:
            logger.debug(f"glob搜索失败: {str(e)}")
        
        return None

    def process_csv_file(self, csv_file_path: Path) -> bool:
        """
        处理CSV文件，添加is_dotnet_binary列
        
        Args:
            csv_file_path: CSV文件路径
            
        Returns:
            bool: 处理是否成功
        """
        try:
            logger.info(f"开始处理CSV文件: {csv_file_path}")
            
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)
            logger.info(f"CSV文件包含 {len(df)} 行数据")
            
            # 检查是否已经存在is_dotnet_binary列
            if 'is_dotnet_binary' in df.columns:
                logger.warning("CSV文件中已存在is_dotnet_binary列，将覆盖现有数据")
            
            # 初始化is_dotnet_binary列
            df['is_dotnet_binary'] = 0
            
            # 统计信息
            total_files = len(df)
            processed_files = 0
            dotnet_files = 0
            error_files = 0
            not_found_files = 0
            
            # 处理每个样本文件
            for index, row in df.iterrows():
                sample_id = row['sample_id']
                
                # 查找样本文件
                sample_file_path = self.find_sample_file(sample_id)
                
                if sample_file_path is None:
                    logger.warning(f"未找到样本文件: {sample_id}")
                    not_found_files += 1
                    continue
                
                # 检查是否为.NET文件
                is_dotnet, error_msg = self.is_dotnet_binary(sample_file_path)
                
                if error_msg:
                    logger.warning(f"处理文件 {sample_id} 时出错: {error_msg}")
                    error_files += 1
                else:
                    processed_files += 1
                    if is_dotnet:
                        df.at[index, 'is_dotnet_binary'] = 1
                        dotnet_files += 1
                
                # 每处理100个文件输出一次进度
                if (index + 1) % 100 == 0:
                    logger.info(f"已处理 {index + 1}/{total_files} 个文件")
            
            # 保存更新后的CSV文件
            output_csv_path = csv_file_path.parent / f"{csv_file_path.stem}_with_dotnet{csv_file_path.suffix}"
            df.to_csv(output_csv_path, index=False)
            
            # 输出统计信息
            logger.info(f"处理完成！")
            logger.info(f"总文件数: {total_files}")
            logger.info(f"成功处理: {processed_files}")
            logger.info(f"未找到文件: {not_found_files}")
            logger.info(f"错误文件: {error_files}")
            logger.info(f".NET文件数: {dotnet_files}")
            if processed_files > 0:
                logger.info(f".NET文件比例: {dotnet_files/processed_files*100:.2f}%")
            logger.info(f"输出文件: {output_csv_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"处理CSV文件时出错: {str(e)}")
            return False
    
    def process_all_csv_files(self) -> bool:
        """
        处理输出目录中的所有CSV文件
        
        Returns:
            bool: 处理是否成功
        """
        try:
            # 查找所有CSV文件
            csv_files = list(self.output_path.glob("*.csv"))
            
            if not csv_files:
                logger.error(f"在 {self.output_path} 中未找到CSV文件")
                return False
            
            logger.info(f"找到 {len(csv_files)} 个CSV文件")
            
            success_count = 0
            for csv_file in csv_files:
                logger.info(f"处理文件: {csv_file.name}")
                if self.process_csv_file(csv_file):
                    success_count += 1
                else:
                    logger.error(f"处理文件 {csv_file.name} 失败")
            
            logger.info(f"成功处理 {success_count}/{len(csv_files)} 个CSV文件")
            return success_count == len(csv_files)
            
        except Exception as e:
            logger.error(f"处理所有CSV文件时出错: {str(e)}")
            return False


def main():
    """
    主函数
    """
    # 设置路径
    base_path = Path(__file__).parent.parent.parent
    dataset_path = base_path / "dataset"
    output_path = base_path / "output"
    
    logger.info("开始.NET二进制文件检测")
    logger.info(f"基础路径: {base_path}")
    
    # 检查路径是否存在
    if not dataset_path.exists():
        logger.error(f"数据集路径不存在: {dataset_path}")
        return False
    
    if not output_path.exists():
        logger.error(f"输出路径不存在: {output_path}")
        return False
    
    # 创建检测器并处理文件
    detector = DotNetDetector(str(dataset_path), str(output_path))
    
    success = detector.process_all_csv_files()
    
    if success:
        logger.info("所有文件处理完成！")
    else:
        logger.error("部分文件处理失败，请查看日志了解详情")
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        sys.exit(1)