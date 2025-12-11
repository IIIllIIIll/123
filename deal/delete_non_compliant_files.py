#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件删除脚本 - 删除JSON中记录的所有不符合规范的文件
支持删除：invalid_pe_files, non_sha256_files, no_dll_pe_files, unanalyzable_files
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any


class FileDeleter:
    def __init__(self, json_file_path: str, target_directory: str):
        """
        初始化文件删除器
        
        Args:
            json_file_path: JSON结果文件路径
            target_directory: 目标文件所在目录
        """
        self.json_file_path = json_file_path
        self.target_directory = Path(target_directory)
        self.deleted_files = []
        self.failed_deletions = []
        
    def load_json_data(self) -> Dict[str, Any]:
        """加载JSON数据"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载JSON文件失败: {e}")
            sys.exit(1)
    
    def collect_files_to_delete(self, data: Dict[str, Any]) -> List[str]:
        """收集需要删除的文件列表"""
        files_to_delete = []
        
        # 收集各类不符合规范的文件
        categories = [
            'invalid_pe_files',
            'non_sha256_files', 
            'no_dll_pe_files',
            'unanalyzable_files'
        ]
        
        for category in categories:
            if category in data and data[category]:
                for file_info in data[category]:
                    filename = file_info.get('filename', '')
                    if filename:
                        files_to_delete.append(filename)
                        
        return files_to_delete
    
    def confirm_deletion(self, files_to_delete: List[str]) -> bool:
        """确认删除操作"""
        print(f"\n📋 发现 {len(files_to_delete)} 个不符合规范的文件需要删除:")
        print(f"📁 目标目录: {self.target_directory}")
        
        # 显示前10个文件作为预览
        preview_count = min(10, len(files_to_delete))
        for i in range(preview_count):
            print(f"   - {files_to_delete[i]}")
        
        if len(files_to_delete) > preview_count:
            print(f"   ... 还有 {len(files_to_delete) - preview_count} 个文件")
        
        print("\n⚠️  警告: 此操作将永久删除这些文件!")
        
        while True:
            response = input("\n是否继续删除? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no', '']:
                return False
            else:
                print("请输入 y 或 n")
    
    def delete_files(self, files_to_delete: List[str]) -> None:
        """执行文件删除"""
        print(f"\n🗑️  开始删除 {len(files_to_delete)} 个文件...")
        
        for i, filename in enumerate(files_to_delete, 1):
            file_path = self.target_directory / filename
            
            try:
                if file_path.exists():
                    file_path.unlink()  # 删除文件
                    self.deleted_files.append(filename)
                    print(f"✅ [{i}/{len(files_to_delete)}] 已删除: {filename}")
                else:
                    print(f"⚠️  [{i}/{len(files_to_delete)}] 文件不存在: {filename}")
                    
            except Exception as e:
                self.failed_deletions.append((filename, str(e)))
                print(f"❌ [{i}/{len(files_to_delete)}] 删除失败: {filename} - {e}")
    
    def print_summary(self) -> None:
        """打印删除结果摘要"""
        print(f"\n📊 删除操作完成!")
        print(f"✅ 成功删除: {len(self.deleted_files)} 个文件")
        
        if self.failed_deletions:
            print(f"❌ 删除失败: {len(self.failed_deletions)} 个文件")
            for filename, error in self.failed_deletions:
                print(f"   - {filename}: {error}")
    
    def run(self) -> None:
        """执行完整的删除流程"""
        print("🚀 文件删除脚本启动")
        print(f"📄 JSON文件: {self.json_file_path}")
        print(f"📁 目标目录: {self.target_directory}")
        
        # 检查目录是否存在
        if not self.target_directory.exists():
            print(f"❌ 目标目录不存在: {self.target_directory}")
            sys.exit(1)
        
        # 加载JSON数据
        data = self.load_json_data()
        
        # 收集需要删除的文件
        files_to_delete = self.collect_files_to_delete(data)
        
        if not files_to_delete:
            print("✅ 没有发现需要删除的不符合规范文件")
            return
        
        # 确认删除
        if not self.confirm_deletion(files_to_delete):
            print("❌ 用户取消删除操作")
            return
        
        # 执行删除
        self.delete_files(files_to_delete)
        
        # 打印摘要
        self.print_summary()


def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python delete_non_compliant_files.py <json_file_path> <target_directory>")
        print("示例: python delete_non_compliant_files.py benign.json D:\\dataset\\benign")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    target_directory = sys.argv[2]
    
    # 检查JSON文件是否存在
    if not os.path.exists(json_file_path):
        print(f"❌ JSON文件不存在: {json_file_path}")
        sys.exit(1)
    
    # 创建删除器并运行
    deleter = FileDeleter(json_file_path, target_directory)
    deleter.run()


if __name__ == "__main__":
    main()