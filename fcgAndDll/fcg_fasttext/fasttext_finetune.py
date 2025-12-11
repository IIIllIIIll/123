#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FastText模型训练工具

使用FastText CLI进行基于预训练词向量的模型训练。

主要功能:
1. 数据预处理和分割
2. 基于预训练词向量的FastText模型训练
3. 模型评估和性能分析
4. 训练报告生成

使用方式:
    python fasttext_finetune.py --pretrained-model cc.en.300.vec --corpus corpus.txt --output-dir ./output

依赖:
    - FastText C++命令行工具
    - Python fasttext库
    - 预训练向量文件（如cc.en.300.vec）
"""

import os
import sys
import argparse
import logging
import subprocess
from typing import List, Tuple, Optional
import fasttext
import numpy as np
from sklearn.model_selection import train_test_split


class FastTextFineTuner:
    """FastText模型训练器"""
    
    def __init__(self, pretrained_model_path: str, corpus_path: str, output_dir: str, model_dim: int):
        """初始化FastText训练器"""
        self.pretrained_model_path = pretrained_model_path
        self.corpus_path = corpus_path
        self.output_dir = output_dir
        self.model_dim = model_dim
        
        os.makedirs(output_dir, exist_ok=True)
        self.setup_logging()
        self.model = None
        
    def setup_logging(self):
        """设置日志配置"""
        log_file = os.path.join(self.output_dir, 'fasttext_finetune.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def preprocess_corpus(self) -> Tuple[List[str], int]:
        """预处理语料库数据"""
        try:
            self.logger.info(f"预处理语料库: {self.corpus_path}")
            
            processed_texts = []
            total_lines = 0
            
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 过滤过短的函数名
                    functions = [func for func in line.split() if len(func) >= 3]
                    if functions:
                        processed_texts.append(' '.join(functions))
                    total_lines += 1
            
            self.logger.info(f"预处理完成，有效文本: {len(processed_texts)} 行")
            return processed_texts, total_lines
            
        except Exception as e:
            self.logger.error(f"预处理失败: {e}")
            return [], 0
    
    def create_training_data(self, texts: List[str], test_size: float = 0.2) -> Tuple[str, str]:
        """创建训练和验证数据文件"""
        try:
            # 分割数据
            if len(texts) > 1:
                train_texts, val_texts = train_test_split(texts, test_size=test_size, random_state=42)
            else:
                train_texts = texts
                val_texts = texts[:1] if texts else []
            
            # 创建文件
            train_file = os.path.join(self.output_dir, 'train_corpus.txt')
            val_file = os.path.join(self.output_dir, 'val_corpus.txt')
            
            with open(train_file, 'w', encoding='utf-8') as f:
                for text in train_texts:
                    f.write(text + '\n')
            
            with open(val_file, 'w', encoding='utf-8') as f:
                for text in val_texts:
                    f.write(text + '\n')
            
            self.logger.info(f"训练数据: {len(train_texts)} 行，验证数据: {len(val_texts)} 行")
            return train_file, val_file
            
        except Exception as e:
            self.logger.error(f"创建训练数据失败: {e}")
            return "", ""
    
    def finetune_model(self, train_file: str, epochs: int = 5, lr: float = 0.05, min_count: int = 1) -> bool:
        """基于预训练词向量训练FastText模型"""
        try:
            self.logger.info("开始训练FastText模型...")
            
            # 定义输出模型前缀
            output_model_prefix = os.path.join(self.output_dir, 'finetuned_fasttext_model')
            finetuned_model_path = f"{output_model_prefix}.bin"

            # 构建CLI命令
            command = [
                'fasttext', 'skipgram',
                '-pretrainedVectors', self.pretrained_model_path,
                '-dim', str(self.model_dim),
                '-output', output_model_prefix,
                '-input', train_file,
                '-epoch', str(epochs),
                '-lr', str(lr),
                '-minCount', str(min_count),
                '-thread', '4',
                '-verbose', '2'
            ]
            
            # 执行命令
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                self.logger.error(f"FastText CLI执行失败: {result.stderr}")
                return False
            
            if not os.path.exists(finetuned_model_path):
                self.logger.error(f"模型文件未生成: {finetuned_model_path}")
                return False

            # 加载训练好的模型
            self.model = fasttext.load_model(finetuned_model_path)
            self.logger.info(f"模型训练完成: {finetuned_model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {e}")
            return False

    def evaluate_model(self, val_file: str) -> dict:
        """评估模型性能 (增强版：计算相邻相似度)"""
        try:
            if not self.model:
                self.logger.error("模型未加载")
                return {}
            
            self.logger.info("正在评估模型...")
            
            # 读取验证数据
            with open(val_file, 'r', encoding='utf-8') as f:
                val_texts = [line.strip() for line in f if line.strip()]
            
            if not val_texts:
                self.logger.warning("验证数据为空")
                return {}
            
            # 计算词汇覆盖率
            total_words = 0
            covered_words = 0
            all_words_set = set()
            
            for text in val_texts:
                words = text.split()
                total_words += len(words)
                all_words_set.update(words)
                for word in words:
                    if word in self.model.words:
                        covered_words += 1
            
            coverage = covered_words / total_words if total_words > 0 else 0
            
            # --- 关键评估：计算相邻函数名的平均余弦相似度 ---
            # 这是 SkipGram 模型的核心评估指标
            similarities = []
            # 抽样评估，避免计算过慢
            sample_texts = val_texts[:min(500, len(val_texts))]
            
            for text in sample_texts:
                functions = text.split()
                if len(functions) >= 2:
                    for i in range(len(functions) - 1):
                        try:
                            # 获取向量
                            vec1 = self.model.get_word_vector(functions[i])
                            vec2 = self.model.get_word_vector(functions[i + 1])
                            
                            # 计算余弦相似度
                            norm1 = np.linalg.norm(vec1)
                            norm2 = np.linalg.norm(vec2)
                            
                            if norm1 > 0 and norm2 > 0:
                                sim = np.dot(vec1, vec2) / (norm1 * norm2)
                                similarities.append(sim)
                        except:
                            # 处理可能的异常情况
                            continue
            
            avg_similarity = np.mean(similarities) if similarities else 0
            
            results = {
                'vocabulary_size': len(self.model.words),
                'dimension': self.model.get_dimension(),
                'word_coverage_rate': coverage,
                'avg_adjacent_similarity': avg_similarity,  # <-- 关键指标
                'total_words_in_val': total_words,
                'total_unique_words_in_val': len(all_words_set),
                'covered_words_in_val': covered_words,
                'similarity_samples_count': len(similarities)
            }
            
            self.logger.info("评估结果:")
            for key, value in results.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"评估失败: {e}")
            return {}
    
    def test_model(self, test_words: List[str] = None) -> dict:
        """测试模型功能 (增强版：包含最近邻测试)"""
        try:
            if not self.model:
                self.logger.error("模型未加载")
                return {}
            
            self.logger.info("正在测试模型...")
            
            # 默认测试词汇
            if not test_words:
                test_words = [
                    'main', 'start', 'sub_401000', 'GetProcAddress', 'CreateFileA',
                    'malloc', 'free', 'printf', 'scanf', 'exit'
                ]
            
            results = {}
            oov_count = 0
            
            for word in test_words:
                try:
                    in_vocab = word in self.model.words
                    if not in_vocab:
                        oov_count += 1
                    
                    word_result = {
                        'in_vocabulary': in_vocab,
                        'vector_norm': float(np.linalg.norm(self.model.get_word_vector(word)))
                    }
                    
                    # --- 增强功能：获取最近邻 ---
                    # 这是评估词向量质量的重要指标
                    if in_vocab:
                        try:
                            # 获取最近邻词汇 (k=5)
                            nearest_neighbors = self.model.get_nearest_neighbors(word, k=5)
                            word_result['nearest_neighbors'] = [
                                {'word': neighbor[1], 'similarity': float(neighbor[0])}
                                for neighbor in nearest_neighbors
                            ]
                        except Exception as e:
                            word_result['nearest_neighbors_error'] = str(e)
                    else:
                        word_result['nearest_neighbors'] = []
                    
                    results[word] = word_result
                    
                except Exception as e:
                    results[word] = {'in_vocabulary': False, 'error': str(e)}
            
            # 统计信息
            total_test_words = len(test_words)
            results['_statistics'] = {
                'total_test_words': total_test_words,
                'oov_words_count': oov_count,
                'oov_rate': oov_count / total_test_words if total_test_words > 0 else 0,
                'in_vocab_words_count': total_test_words - oov_count
            }
            
            self.logger.info(f"测试完成 - OOV率: {results['_statistics']['oov_rate']:.2%}")
            self.logger.info(f"词汇表内词汇数: {results['_statistics']['in_vocab_words_count']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"测试失败: {e}")
            return {}
    
    def run_finetune(self, epochs: int = 5, lr: float = 0.05, min_count: int = 1, test_size: float = 0.2) -> bool:
        """运行完整的训练流程"""
        try:
            self.logger.info("开始FastText模型训练流程")
            
            # 1. 预处理语料库
            texts, _ = self.preprocess_corpus()
            if not texts:
                self.logger.error("没有有效的训练数据")
                return False
            
            # 2. 创建训练数据
            train_file, val_file = self.create_training_data(texts, test_size)
            if not train_file or not val_file:
                return False
            
            # 3. 训练模型
            if not self.finetune_model(train_file, epochs, lr, min_count):
                return False
            
            # 4. 评估和测试
            eval_results = self.evaluate_model(val_file)
            test_results = self.test_model()
            
            # 5. 保存报告
            self.save_report(eval_results, test_results, {'epochs': epochs, 'lr': lr, 'min_count': min_count})
            
            self.logger.info("FastText模型训练完成！")
            return True
            
        except Exception as e:
            self.logger.error(f"训练流程失败: {e}")
            return False
    
    def save_report(self, eval_results: dict, test_results: dict, train_params: dict = None):
        """保存训练报告"""
        try:
            report_file = os.path.join(self.output_dir, 'finetune_report.txt')
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("FastText模型训练报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("配置信息:\n")
                f.write(f"  预训练模型: {self.pretrained_model_path}\n")
                f.write(f"  语料库: {self.corpus_path}\n")
                f.write(f"  输出目录: {self.output_dir}\n\n")
                
                if train_params:
                    f.write("训练参数:\n")
                    for key, value in train_params.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                f.write("评估结果:\n")
                for key, value in eval_results.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
                
                f.write("测试结果:\n")
                for word, result in test_results.items():
                    f.write(f"  {word}: {result}\n")
            
            self.logger.info(f"报告已保存: {report_file}")
            
        except Exception as e:
            self.logger.error(f"保存报告失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FastText模型训练工具')
    
    parser.add_argument('--pretrained-model', 
                       default='/mnt/data1_l20_raid5disk/lbq_dataset/models/crawl-300d-2M-subword.vec',
                       help='预训练向量文件路径 (需要.vec文件)')
    
    parser.add_argument('--corpus',
                       default='/mnt/data1_l20_raid5disk/lbq_dataset/output/fcg_corpus.txt',
                       help='语料库文件路径')
    
    parser.add_argument('--output-dir',
                       default='/mnt/data1_l20_raid5disk/lbq_dataset/output/fcg_fasttext',
                       help='输出目录')
    
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数 (默认: 5)')
    parser.add_argument('--lr', type=float, default=0.05, help='学习率 (默认: 0.05)')
    parser.add_argument('--min-count', type=int, default=1, help='最小词频 (默认: 1)')
    parser.add_argument('--test-size', type=float, default=0.2, help='验证集比例 (默认: 0.2)')
    parser.add_argument('--dim', type=int, default=300, help='向量维度 (默认: 300)')
    
    args = parser.parse_args()
    
    # 检查FastText CLI工具
    try:
        subprocess.run(['fasttext'], capture_output=True, text=True)
        print("✅ FastText CLI工具检测成功")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 错误: FastText CLI工具未找到")
        sys.exit(1)
    
    # 检查文件存在性
    if not os.path.exists(args.pretrained_model):
        print(f"错误: 预训练向量文件不存在: {args.pretrained_model}")
        sys.exit(1)
    
    if not os.path.exists(args.corpus):
        print(f"错误: 语料库文件不存在: {args.corpus}")
        sys.exit(1)
    
    # 创建训练器并运行
    finetuner = FastTextFineTuner(
        pretrained_model_path=args.pretrained_model,
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        model_dim=args.dim
    )
    
    success = finetuner.run_finetune(
        epochs=args.epochs,
        lr=args.lr,
        min_count=args.min_count,
        test_size=args.test_size
    )
    
    if success:
        print("✅ FastText模型训练成功完成！")
        print(f"📁 输出目录: {args.output_dir}")
        sys.exit(0)
    else:
        print("❌ FastText模型训练失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()