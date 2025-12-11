#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
malware api 分类工具、list
API分类工具
用于比较JSON文件中的API列表与TXT文件中的API列表，找出差异
并按照malware_api分类对API进行归类，生成分类结果

作者: API分类工具
创建时间: 2025-01-16
更新时间: 2025-01-16
"""

import json
import os
import sys
from typing import Set, List, Dict, Any
from datetime import datetime


class APIClassifier:
    """API分类器类"""
    
    def __init__(self):
        """初始化API分类器"""
        self.json_apis: Set[str] = set()
        self.malware_categories: Dict[str, Set[str]] = {}
        self.classification_results: Dict[str, List[str]] = {}
        
    def load_malware_categories(self, malware_api_dir: str) -> bool:
        """
        从malware_api目录加载所有分类文件
        
        Args:
            malware_api_dir: malware_api目录路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(malware_api_dir):
                print(f"错误: malware_api目录不存在 - {malware_api_dir}")
                return False
            
            # 定义分类文件映射
            category_files = {
                'debug': 'debug_malapi.txt',
                'enum': 'enum_malapi.txt', 
                'evade': 'evade_malapi.txt',
                'helper': 'helper_malapi.txt',
                'inet': 'inet_malapi.txt',
                'injection': 'injetion_malapi.txt',  # 注意原文件名拼写
                'ransom': 'ransom_malapi.txt',
                'spy': 'spy_malapi.txt'
            }
            
            for category, filename in category_files.items():
                file_path = os.path.join(malware_api_dir, filename)
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        apis = set()
                        for line in f:
                            api = line.strip()
                            if api:  # 忽略空行
                                apis.add(api.lower())
                        self.malware_categories[category] = apis
                        print(f"成功加载 {category} 分类，包含 {len(apis)} 个API")
                else:
                    print(f"警告: 分类文件不存在 - {file_path}")
            
            print(f"总共加载了 {len(self.malware_categories)} 个API分类")
            return len(self.malware_categories) > 0
            
        except Exception as e:
            print(f"错误: 加载malware_api分类时发生异常 - {e}")
            return False
        
    def load_json_apis(self, json_file_path: str) -> bool:
        """
        从JSON文件中加载API列表
        
        Args:
            json_file_path: JSON文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(json_file_path):
                print(f"错误: JSON文件不存在 - {json_file_path}")
                return False
                
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 从JSON结构中提取API列表
            if 'apis' in data and 'unique_list' in data['apis']:
                self.json_apis = set(api.lower() for api in data['apis']['unique_list'])
                print(f"成功从JSON文件加载 {len(self.json_apis)} 个API")
                return True
            else:
                print("错误: JSON文件格式不正确，缺少 'apis.unique_list' 字段")
                return False
                
        except json.JSONDecodeError as e:
            print(f"错误: JSON文件解析失败 - {e}")
            return False
        except Exception as e:
            print(f"错误: 读取JSON文件时发生异常 - {e}")
            return False
    
    def classify_apis(self) -> Dict[str, List[str]]:
        """
        对JSON中的API进行分类
        
        Returns:
            Dict[str, List[str]]: 分类结果，键为分类名，值为API列表
        """
        # 初始化分类结果
        self.classification_results = {category: [] for category in self.malware_categories.keys()}
        self.classification_results['unclassified'] = []
        
        # 对每个API进行分类
        for api in self.json_apis:
            classified = False
            
            # 检查API属于哪个分类
            for category, category_apis in self.malware_categories.items():
                if api in category_apis:
                    self.classification_results[category].append(api)
                    classified = True
                    break  # API只归类到第一个匹配的分类
            
            # 如果没有找到匹配的分类，归为未分类
            if not classified:
                self.classification_results['unclassified'].append(api)
        
        # 对每个分类的API进行排序
        for category in self.classification_results:
            self.classification_results[category].sort()
        
        return self.classification_results
    
    def generate_classification_report(self) -> Dict[str, Any]:
        """
        生成分类报告
        
        Returns:
            Dict[str, Any]: 包含分类统计和详细信息的报告
        """
        report = {
            "classification_info": {
                "classification_time": datetime.now().isoformat(),
                "total_apis_processed": len(self.json_apis),
                "total_categories": len(self.malware_categories),
                "classified_apis": sum(len(apis) for category, apis in self.classification_results.items() if category != 'unclassified'),
                "unclassified_apis": len(self.classification_results.get('unclassified', []))
            },
            "categories": {}
        }
        
        # 添加每个分类的详细信息
        for category, apis in self.classification_results.items():
            if category != 'unclassified':
                report["categories"][category] = {
                    "count": len(apis),
                    "apis": apis,
                    "description": self._get_category_description(category)
                }
        
        # 添加未分类的API
        if self.classification_results.get('unclassified'):
            report["categories"]["unclassified"] = {
                "count": len(self.classification_results['unclassified']),
                "apis": self.classification_results['unclassified'],
                "description": "未能归类到任何已知恶意软件API分类的API"
            }
        
        return report
    
    def _get_category_description(self, category: str) -> str:
        """
        获取分类描述
        
        Args:
            category: 分类名称
            
        Returns:
            str: 分类描述
        """
        descriptions = {
            'debug': '调试和反调试相关API',
            'enum': '系统枚举和信息收集API',
            'evade': '逃避检测和隐藏行为API',
            'helper': '辅助功能和系统操作API',
            'inet': '网络通信和互联网访问API',
            'injection': '代码注入和进程操作API',
            'ransom': '加密和勒索软件相关API',
            'spy': '键盘记录和监控相关API'
        }
        return descriptions.get(category, f'{category}分类API')
    
    def save_classification_results(self, output_file: str) -> bool:
        """
        保存分类结果到JSON文件
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            bool: 保存是否成功
        """
        try:
            report = self.generate_classification_report()
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            print(f"分类结果已保存到: {output_file}")
            return True
            
        except Exception as e:
            print(f"错误: 保存分类结果时发生异常 - {e}")
            return False
    
    def print_classification_summary(self):
        """打印分类摘要到控制台"""
        print("\n" + "="*80)
        print("API分类结果摘要")
        print("="*80)
        
        total_apis = len(self.json_apis)
        classified_count = sum(len(apis) for category, apis in self.classification_results.items() if category != 'unclassified')
        unclassified_count = len(self.classification_results.get('unclassified', []))
        
        print(f"\n📊 总体统计:")
        print(f"   处理的API总数:     {total_apis}")
        print(f"   已分类API数量:     {classified_count}")
        print(f"   未分类API数量:     {unclassified_count}")
        print(f"   分类覆盖率:       {classified_count/total_apis*100:.1f}%")
        
        print(f"\n📋 各分类详情:")
        print("-" * 60)
        
        # 按API数量排序显示
        sorted_categories = sorted(
            [(cat, apis) for cat, apis in self.classification_results.items() if apis],
            key=lambda x: len(x[1]), reverse=True
        )
        
        for category, apis in sorted_categories:
            description = self._get_category_description(category) if category != 'unclassified' else '未分类API'
            print(f"   {category:12s} | {len(apis):3d} 个API | {description}")
        
        print("\n" + "="*80)
    
    # 保留原有的APIComparator类以保持向后兼容性
class APIComparator:
    """API比较器类（保留用于向后兼容）"""
    
    def __init__(self):
        """初始化API比较器"""
        self.json_apis: Set[str] = set()
        self.txt_apis: Set[str] = set()
        
    def load_json_apis(self, json_file_path: str) -> bool:
        """从JSON文件中加载API列表"""
        try:
            if not os.path.exists(json_file_path):
                print(f"错误: JSON文件不存在 - {json_file_path}")
                return False
                
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'apis' in data and 'unique_list' in data['apis']:
                self.json_apis = set(api.lower() for api in data['apis']['unique_list'])
                print(f"成功从JSON文件加载 {len(self.json_apis)} 个API")
                return True
            else:
                print("错误: JSON文件格式不正确，缺少 'apis.unique_list' 字段")
                return False
                
        except json.JSONDecodeError as e:
            print(f"错误: JSON文件解析失败 - {e}")
            return False
        except Exception as e:
            print(f"错误: 读取JSON文件时发生异常 - {e}")
            return False
    
    def load_txt_apis(self, txt_file_path: str) -> bool:
        """
        从TXT文件中加载API列表
        
        Args:
            txt_file_path: TXT文件路径
            
        Returns:
            bool: 加载是否成功
        """
        try:
            if not os.path.exists(txt_file_path):
                print(f"错误: TXT文件不存在 - {txt_file_path}")
                return False
                
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 清理并转换为小写，去除空行
            self.txt_apis = set()
            for line in lines:
                api = line.strip()
                if api:  # 忽略空行
                    self.txt_apis.add(api.lower())
                    
            print(f"成功从TXT文件加载 {len(self.txt_apis)} 个API")
            return True
            
        except Exception as e:
            print(f"错误: 读取TXT文件时发生异常 - {e}")
            return False
    
    def find_json_unique_apis(self) -> List[str]:
        """
        找出JSON中独有的API（不在TXT中的API）
        
        Returns:
            List[str]: JSON中独有的API列表
        """
        unique_apis = self.json_apis - self.txt_apis
        return sorted(list(unique_apis))
    
    def find_txt_unique_apis(self) -> List[str]:
        """
        找出TXT中独有的API（不在JSON中的API）
        
        Returns:
            List[str]: TXT中独有的API列表
        """
        unique_apis = self.txt_apis - self.json_apis
        return sorted(list(unique_apis))
    
    def find_common_apis(self) -> List[str]:
        """
        找出两个文件中共同的API
        
        Returns:
            List[str]: 共同的API列表
        """
        common_apis = self.json_apis & self.txt_apis
        return sorted(list(common_apis))
    
    def print_comparison_results(self):
        """打印比较结果到控制台"""
        print("\n" + "="*80)
        print("API比较结果")
        print("="*80)
        
        # 统计信息
        json_unique = self.find_json_unique_apis()
        txt_unique = self.find_txt_unique_apis()
        common = self.find_common_apis()
        
        print(f"\n📊 统计信息:")
        print(f"   JSON文件中的API总数: {len(self.json_apis)}")
        print(f"   TXT文件中的API总数:  {len(self.txt_apis)}")
        print(f"   共同API数量:         {len(common)}")
        print(f"   JSON独有API数量:     {len(json_unique)}")
        print(f"   TXT独有API数量:      {len(txt_unique)}")
        
        # JSON中独有的API
        if json_unique:
            print(f"\n🔍 JSON中独有的API ({len(json_unique)}个):")
            print("-" * 50)
            for i, api in enumerate(json_unique, 1):
                print(f"   {i:3d}. {api}")
        else:
            print(f"\n✅ JSON中没有独有的API（所有API都在TXT中存在）")
        
        # TXT中独有的API（可选显示）
        if txt_unique:
            print(f"\n📝 TXT中独有的API ({len(txt_unique)}个):")
            print("-" * 50)
            # 只显示前20个，避免输出过长
            display_count = min(20, len(txt_unique))
            for i, api in enumerate(txt_unique[:display_count], 1):
                print(f"   {i:3d}. {api}")
            if len(txt_unique) > display_count:
                print(f"   ... 还有 {len(txt_unique) - display_count} 个API未显示")
        
        print("\n" + "="*80)


def main():
    """主函数"""
    print("API分类工具")
    print("用于对JSON文件中的API进行恶意软件分类")
    print("-" * 50)
    
    # 文件路径
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    json_file = os.path.join(base_dir, "output", "unique_apis_20251016_170609.json")
    malware_api_dir = os.path.join(base_dir, "output", "malware_api")
    output_file = os.path.join(base_dir, "output", "malware_api_list.json")
    
    print(f"JSON文件路径:        {json_file}")
    print(f"malware_api目录:     {malware_api_dir}")
    print(f"输出文件路径:        {output_file}")
    
    # 创建分类器实例
    classifier = APIClassifier()
    
    # 加载文件
    print("\n正在加载文件...")
    
    # 加载malware_api分类
    if not classifier.load_malware_categories(malware_api_dir):
        print("malware_api分类加载失败，程序退出")
        sys.exit(1)
    
    # 加载JSON文件中的API
    if not classifier.load_json_apis(json_file):
        print("JSON文件加载失败，程序退出")
        sys.exit(1)
    
    # 执行API分类
    print("\n正在进行API分类...")
    classification_results = classifier.classify_apis()
    
    # 打印分类摘要
    classifier.print_classification_summary()
    
    # 保存分类结果
    print(f"\n正在保存分类结果...")
    if classifier.save_classification_results(output_file):
        print("✅ API分类完成！")
    else:
        print("❌ 保存分类结果失败")
        sys.exit(1)


if __name__ == "__main__":
    main()