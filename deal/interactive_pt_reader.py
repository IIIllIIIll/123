#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式PT文件读取器
提供友好的交互界面来浏览和读取PT文件
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))
from pt_file_reader import PTFileReader


class InteractivePTReader:
    """交互式PT文件读取器"""
    
    def __init__(self, data_dir: str = None):
        """初始化交互式读取器"""
        self.reader = PTFileReader(data_dir)
        self.files = self.reader.list_pt_files()
        self.selected_files = []
    
    def display_menu(self):
        """显示主菜单"""
        print("\n" + "="*60)
        print("🔍 PT文件读取器 - 交互式界面")
        print("="*60)
        print(f"📁 数据文件夹: {self.reader.data_dir}")
        print(f"📊 总文件数: {len(self.files)}")
        print(f"✅ 已选择: {len(self.selected_files)} 个文件")
        print("-"*60)
        print("1. 📋 浏览所有文件")
        print("2. 🔍 搜索文件")
        print("3. ➕ 选择文件")
        print("4. ➖ 取消选择")
        print("5. 📖 读取选中的文件")
        print("6. 📊 显示文件摘要")
        print("7. 💾 导出摘要到JSON")
        print("8. 🧹 清空选择")
        print("0. 🚪 退出")
        print("-"*60)
    
    def display_files(self, files: List[str], page_size: int = 20, show_index: bool = True):
        """分页显示文件列表"""
        if not files:
            print("❌ 没有找到文件")
            return
        
        total_pages = (len(files) + page_size - 1) // page_size
        current_page = 0
        
        while True:
            start_idx = current_page * page_size
            end_idx = min(start_idx + page_size, len(files))
            
            print(f"\n📄 第 {current_page + 1}/{total_pages} 页 (显示 {start_idx + 1}-{end_idx}/{len(files)})")
            print("-" * 80)
            
            for i in range(start_idx, end_idx):
                file = files[i]
                status = "✅" if file in self.selected_files else "⭕"
                if show_index:
                    print(f"{status} {i + 1:4d}. {file}")
                else:
                    print(f"{status} {file}")
            
            if total_pages > 1:
                print("-" * 80)
                print("导航: [n]下一页 [p]上一页 [q]返回主菜单")
                choice = input("请选择: ").strip().lower()
                
                if choice == 'n' and current_page < total_pages - 1:
                    current_page += 1
                elif choice == 'p' and current_page > 0:
                    current_page -= 1
                elif choice == 'q':
                    break
                else:
                    if choice not in ['n', 'p']:
                        print("❌ 无效选择")
            else:
                input("\n按回车键返回主菜单...")
                break
    
    def search_files(self):
        """搜索文件"""
        pattern = input("🔍 请输入搜索关键词: ").strip()
        if not pattern:
            print("❌ 搜索关键词不能为空")
            return
        
        matched_files = self.reader.search_files_by_pattern(pattern)
        print(f"\n🎯 搜索 '{pattern}' 找到 {len(matched_files)} 个文件:")
        
        if matched_files:
            self.display_files(matched_files, show_index=False)
        else:
            print("❌ 没有找到匹配的文件")
    
    def select_files(self):
        """选择文件"""
        print("\n📋 选择文件 (输入文件编号，用逗号分隔，或输入范围如1-10)")
        print("💡 提示: 输入 'all' 选择所有文件，'clear' 清空选择")
        
        # 显示前20个文件作为参考
        print("\n前20个文件:")
        for i, file in enumerate(self.files[:20], 1):
            status = "✅" if file in self.selected_files else "⭕"
            print(f"{status} {i:4d}. {file}")
        
        if len(self.files) > 20:
            print(f"... 还有 {len(self.files) - 20} 个文件")
        
        selection = input("\n请输入选择: ").strip()
        
        if selection.lower() == 'all':
            self.selected_files = self.files.copy()
            print(f"✅ 已选择所有 {len(self.files)} 个文件")
        elif selection.lower() == 'clear':
            self.selected_files.clear()
            print("🧹 已清空所有选择")
        else:
            try:
                indices = self._parse_selection(selection)
                for idx in indices:
                    if 1 <= idx <= len(self.files):
                        file = self.files[idx - 1]
                        if file not in self.selected_files:
                            self.selected_files.append(file)
                            print(f"✅ 已添加: {file}")
                        else:
                            print(f"⚠️  已存在: {file}")
                    else:
                        print(f"❌ 无效索引: {idx}")
            except ValueError as e:
                print(f"❌ 输入格式错误: {e}")
    
    def _parse_selection(self, selection: str) -> List[int]:
        """解析选择输入"""
        indices = []
        parts = selection.split(',')
        
        for part in parts:
            part = part.strip()
            if '-' in part:
                # 范围选择
                start, end = part.split('-', 1)
                start_idx = int(start.strip())
                end_idx = int(end.strip())
                indices.extend(range(start_idx, end_idx + 1))
            else:
                # 单个选择
                indices.append(int(part))
        
        return indices
    
    def deselect_files(self):
        """取消选择文件"""
        if not self.selected_files:
            print("❌ 没有已选择的文件")
            return
        
        print(f"\n📋 当前已选择 {len(self.selected_files)} 个文件:")
        for i, file in enumerate(self.selected_files, 1):
            print(f"{i:4d}. {file}")
        
        selection = input("\n请输入要取消选择的编号 (用逗号分隔): ").strip()
        
        try:
            indices = [int(x.strip()) for x in selection.split(',') if x.strip()]
            removed_files = []
            
            for idx in sorted(indices, reverse=True):
                if 1 <= idx <= len(self.selected_files):
                    removed_file = self.selected_files.pop(idx - 1)
                    removed_files.append(removed_file)
                else:
                    print(f"❌ 无效索引: {idx}")
            
            for file in removed_files:
                print(f"➖ 已移除: {file}")
                
        except ValueError:
            print("❌ 输入格式错误")
    
    def read_selected_files(self):
        """读取选中的文件"""
        if not self.selected_files:
            print("❌ 没有选择任何文件")
            return
        
        print(f"\n📖 正在读取 {len(self.selected_files)} 个文件...")
        
        results = self.reader.load_multiple_files(self.selected_files)
        
        print("\n📊 读取结果:")
        print("="*80)
        
        success_count = 0
        for filename, result in results.items():
            if 'error' in result:
                print(f"❌ {filename}")
                print(f"   错误: {result['error']}")
            else:
                success_count += 1
                print(f"✅ {filename}")
                print(f"   大小: {result['file_size']:,} bytes")
                if 'num_nodes' in result:
                    print(f"   节点数: {result['num_nodes']:,}")
                    print(f"   边数: {result['num_edges']:,}")
                    print(f"   特征维度: {result['node_features_dim']}")
            print("-" * 40)
        
        print(f"\n📈 统计: 成功 {success_count}/{len(self.selected_files)} 个文件")
    
    def show_summary(self):
        """显示文件摘要"""
        limit_input = input("📊 请输入要显示的文件数量限制 (回车显示所有): ").strip()
        limit = None
        
        if limit_input:
            try:
                limit = int(limit_input)
            except ValueError:
                print("❌ 无效数字，将显示所有文件")
        
        print(f"\n📊 正在生成摘要...")
        summaries = self.reader.batch_summary(limit)
        
        print(f"\n📋 文件摘要 (共 {len(summaries)} 个文件):")
        print("="*80)
        
        success_count = 0
        total_nodes = 0
        total_edges = 0
        
        for summary in summaries:
            if 'error' in summary:
                print(f"❌ {summary['filename']}: {summary['error']}")
            else:
                success_count += 1
                print(f"✅ {summary['filename']}")
                print(f"   大小: {summary['file_size']:,} bytes")
                if 'num_nodes' in summary:
                    nodes = summary['num_nodes']
                    edges = summary['num_edges']
                    total_nodes += nodes
                    total_edges += edges
                    print(f"   节点: {nodes:,}, 边: {edges:,}, 特征维度: {summary['node_features_dim']}")
            print("-" * 40)
        
        print(f"\n📈 总计统计:")
        print(f"   成功文件: {success_count}/{len(summaries)}")
        print(f"   总节点数: {total_nodes:,}")
        print(f"   总边数: {total_edges:,}")
    
    def export_summary(self):
        """导出摘要到JSON"""
        output_file = input("💾 请输入输出文件名 (默认: pt_files_summary.json): ").strip()
        if not output_file:
            output_file = "pt_files_summary.json"
        
        if not output_file.endswith('.json'):
            output_file += '.json'
        
        limit_input = input("📊 请输入文件数量限制 (回车处理所有文件): ").strip()
        limit = None
        
        if limit_input:
            try:
                limit = int(limit_input)
            except ValueError:
                print("❌ 无效数字，将处理所有文件")
        
        try:
            self.reader.export_summary_to_json(output_file, limit)
            print(f"✅ 摘要已成功导出到: {output_file}")
        except Exception as e:
            print(f"❌ 导出失败: {e}")
    
    def clear_selection(self):
        """清空选择"""
        if self.selected_files:
            self.selected_files.clear()
            print("🧹 已清空所有选择")
        else:
            print("ℹ️  没有已选择的文件")
    
    def run(self):
        """运行交互式界面"""
        print("🚀 启动PT文件读取器...")
        
        while True:
            try:
                self.display_menu()
                choice = input("请选择操作 (0-8): ").strip()
                
                if choice == '0':
                    print("👋 再见!")
                    break
                elif choice == '1':
                    self.display_files(self.files)
                elif choice == '2':
                    self.search_files()
                elif choice == '3':
                    self.select_files()
                elif choice == '4':
                    self.deselect_files()
                elif choice == '5':
                    self.read_selected_files()
                elif choice == '6':
                    self.show_summary()
                elif choice == '7':
                    self.export_summary()
                elif choice == '8':
                    self.clear_selection()
                else:
                    print("❌ 无效选择，请输入 0-8")
                
                if choice != '0':
                    input("\n按回车键继续...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，再见!")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                input("按回车键继续...")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='交互式PT文件读取器')
    parser.add_argument('--data-dir', type=str, help='数据文件夹路径')
    
    args = parser.parse_args()
    
    try:
        app = InteractivePTReader(args.data_dir)
        app.run()
    except Exception as e:
        print(f"❌ 启动失败: {e}")


if __name__ == "__main__":
    main()