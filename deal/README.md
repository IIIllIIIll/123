# PE文件处理工具集

## 功能概述

用于批量处理PE文件的Python脚本工具集，主要用于恶意软件分析和数据集预处理。

## 工具列表

### 1. pe_analyzer.py - PE文件分析器
批量分析PE文件，生成详细的分析报告
- 检测无效PE文件和无DLL导入的PE文件
- 检查非SHA256命名的文件
- 生成JSON格式的分析结果

### 2. file_renamer.py - 文件重命名器
将文件重命名为SHA256哈希值格式
- SHA256重命名功能
- 重复文件检测和处理
- 多线程并行处理

### 3. delete_non_compliant_files.py - 不符合规范文件删除器
根据分析结果删除不符合规范的文件
- 支持删除多种类型的不符合规范文件
- 安全确认机制，防止误删

### 4. data_preparation.py - 数据准备与组织器
机器学习数据集的准备和组织工具
- 自动扫描良性和恶意样本目录
- 按比例划分训练集、测试集、验证集（7:2:1）
- 生成标准化的labels.csv文件

### 5. random_sample_copier.py - 随机样本复制器
从大型数据集中随机选择样本进行复制
- 随机选择指定数量的文件
- 支持自定义源目录和目标目录

### 6. api_comparator.py - API分类与比较工具
API分类和比较分析工具
- 比较JSON文件中的API列表与TXT文件中的API列表
- 按照malware_api分类对API进行归类
- 支持多种恶意软件API分类（debug、enum、evade、helper、inet、injection、ransom、spy）

### 7. add_dotnet_label.py - .NET二进制检测器
为CSV文件添加.NET二进制标签
- 检测PE文件是否为.NET二进制文件
- 基于mscoree.dll导入判断.NET特征
- 为数据集CSV文件添加is_dotnet_binary标签

### 8. pt_file_reader.py - PT文件读取器
用于读取和分析PyTorch图数据文件(.pt)的核心工具
- 支持单个或批量读取PT文件
- 提取图数据的基本信息（节点数、边数、特征维度等）
- 文件搜索和过滤功能
- 导出摘要信息到JSON格式

### 9. interactive_pt_reader.py - 交互式PT文件读取器
提供友好的交互式命令行界面来浏览和读取PT文件
- 分页浏览文件列表
- 交互式文件选择和批量操作
- 实时搜索和过滤
- 图形化菜单界面

## 使用方法

### PE文件分析器
```bash
python pe_analyzer.py
```

### 文件删除器
```bash
python delete_non_compliant_files.py <json_file_path> <target_directory>
```

### 文件重命名器
```bash
python file_renamer.py
```

### 数据准备与组织器
```bash
python data_preparation.py
```

### 随机样本复制器
```bash
python random_sample_copier.py
```

### API分类与比较工具
```bash
python api_comparator.py
```

### .NET二进制检测器
```bash
python add_dotnet_label.py
```

## 工作流程建议

1. **数据准备**: 使用 `data_preparation.py` 组织数据集
2. **文件分析**: 使用 `pe_analyzer.py` 分析PE文件
3. **数据清理**: 使用 `delete_non_compliant_files.py` 清理不符合规范的文件
4. **文件标准化**: 使用 `file_renamer.py` 重命名文件为SHA256格式
5. **.NET标签**: 使用 `add_dotnet_label.py` 添加.NET标签
6. **API分析**: 使用 `api_comparator.py` 进行API分类分析

## 依赖库

```bash
pip install pefile pandas
```

## 注意事项

1. **备份重要数据**: 运行前请备份重要文件
2. **权限要求**: 确保对目标目录有读写权限
3. **磁盘空间**: 确保有足够的磁盘空间进行文件操作
4. **文件删除**: 删除操作不可恢复，请谨慎操作