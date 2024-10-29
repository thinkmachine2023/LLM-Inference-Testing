#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Test Prompt Generator
~~~~~~~~~~~~~~~~~~~~~~~~

A tool for generating test prompts with different input/output token patterns.

Patterns:
    1. Short Input / Long Output  (e.g., text generation, story writing)
    2. Long Input / Long Output   (e.g., text analysis, detailed summary)
    3. Long Input / Short Output  (e.g., classification, sentiment analysis)

Author: Liang Zhun
Version: 1.0.0
Created: October 2024
License: Apache License 2.0
"""

import pandas as pd
import numpy as np
from modelscope import AutoTokenizer
from typing import List, Dict
import json

class PromptGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('qwen/Qwen-7B-Chat', trust_remote_code=True)
        
        # 扩展内容库
        self.content_library = {

                # 短输入模板
    'short_input': [
        "请写一篇关于{topic}的详细技术报告，要求全面分析其发展现状、关键技术、应用场景和未来趋势",
        "详细分析{topic}在未来3-5年的发展趋势，包括技术创新、市场规模、应用领域等方面",
        "请从技术架构、性能指标、应用效果等方面，深入介绍{topic}的最新进展",
        "对{topic}进行全面的行业分析，包括市场格局、竞争态势、投资机会等方面",
        "请详细讨论{topic}在各个行业的具体应用案例，并分析其实施效果和经验教训",
        "分析{topic}面临的主要挑战和解决方案，包括技术、成本、人才等方面的问题",
        "请详细介绍{topic}的核心技术原理和创新特点，并评估其商业价值",
        "对{topic}的产业生态进行深入分析，包括上下游关系、商业模式、发展机遇等",
        "详细阐述{topic}在智慧城市建设中的应用价值和实施路径",
        "分析{topic}在数字化转型中的关键作用，并提供具体的落地建议"
    ],
            # 主题
            'topics': [
                "人工智能", "量子计算", "区块链", "5G技术",
                "自动驾驶", "元宇宙", "物联网", "绿色能源"
            ],
            
            # 背景介绍
            'background': [
                "该领域近年来发展迅速，投资规模持续扩大。根据最新统计，全球市场规模已超过万亿美元，年增长率保持在30%以上。主要发达国家都将其列为重点发展方向，并投入大量研发资源。",
                "作为新兴技术领域的重要分支，该方向受到学术界和产业界的广泛关注。近五年来，相关研究论文数量增长了300%，专利申请数量更是翻了四倍。多个国际科技巨头也相继布局。",
                "在数字化转型的大背景下，该技术已成为产业升级的关键推动力。企业对相关解决方案的需求持续增长，市场规模预计在2025年将达到2万亿美元。"
            ],
            
            # 主要内容
            'main_content': [
                "从技术发展来看，主要呈现出三个特点：第一，技术创新速度加快，新的突破持续涌现；第二，应用场景不断拓展，已渗透到各个行业；第三，产业生态逐步成熟，形成了完整的供应链。",
                "行业发展呈现出明显的区域性特征。北美地区在基础研究和技术创新上保持领先；亚太地区在应用落地和市场规模上增长最快；欧洲在标准制定和行业规范上贡献显著。",
                "从市场结构看，分为三个主要层次：底层基础设施提供商、中间件服务商和应用解决方案提供商。市场集中度较高，头部企业占据了80%的市场份额。"
            ],
            
            # 技术细节
            'technical_details': [
                "核心技术架构包括：1. 分布式计算框架，支持大规模并行处理；2. 数据处理引擎，实现实时分析；3. 智能决策系统，提供自适应优化。系统性能达到业界领先水平，平均响应时间小于10ms。",
                "关键技术指标：处理延迟<100ms，准确率>99.9%，系统吞吐量>10000QPS。采用最新的微服务架构，支持容器化部署，具备强大的扩展能力和容错能力。",
                "技术实现基于云原生架构，采用容器化和微服务设计，支持多云部署和动态扩缩容。系统可靠性达到99.999%，年故障停机时间小于5分钟。"
            ],
            
            # 应用场景
            'applications': [
                "在智能制造领域，该技术已成功应用于生产线优化、质量控制和预测性维护，帮助企业提升生产效率30%以上，减少设备故障率50%。在金融领域，用于风险控制和智能投顾，准确率达到95%。",
                "典型应用场景包括：智慧城市建设、工业自动化、医疗诊断、金融风控等。其中，在智慧城市项目中，已实现交通流量优化、能源管理和环境监测等功能，效益显著。",
                "目前已在多个行业实现规模化应用，包括：制造业的智能生产线、金融业的风险控制系统、医疗行业的辅助诊断系统等。使用该技术的企业平均效率提升40%，成本降低25%。"
            ],
            
            # 发展趋势
            'trends': [
                "未来发展趋势主要体现在三个方面：1. 技术融合加深，与AI、5G等新技术深度结合；2. 应用场景扩展，向更多垂直领域渗透；3. 产业链整合，形成更完善的生态系统。",
                "预计在未来3-5年内，该领域将迎来几个重要突破：算法效率提升10倍、能耗降低50%、部署成本降低60%。这些进展将极大推动行业发展。",
                "行业发展呈现出明显的趋势：技术持续创新、应用不断深化、生态日益完善。预计到2025年，市场规模将突破5万亿美元，年均增长率保持在40%以上。"
            ],
            
            # 问题与挑战
            'challenges': [
                "当前面临的主要挑战包括：1. 技术标准尚未统一，制约了行业发展；2. 成本仍然较高，影响推广速度；3. 人才供给不足，特别是高端人才稀缺；4. 安全性问题需要进一步解决。",
                "存在的问题主要有：技术成熟度有待提高、商业模式需要验证、行业标准不统一、人才储备不足等。这些问题需要产业链各方共同努力解决。",
                "主要挑战体现在：技术创新难度加大、应用成本居高不下、人才供给不足、安全风险增加等方面。解决这些问题需要长期投入和行业协作。"
            ],
            
            # 市场分析相关内容
            'market_background': [
                "市场规模持续扩大，2023年全球市场规模达到1.5万亿美元，预计2025年将突破3万亿美元。主要市场分布在北美（40%）、亚太（35%）和欧洲（20%）。",
                "行业呈现快速增长态势，近三年复合增长率达到45%。投资热度持续上升，2023年全球相关投资超过1000亿美元，创历史新高。",
                "市场需求旺盛，企业数字化转型推动行业快速发展。头部企业市占率超过60%，行业集中度进一步提升。"
            ],
            
            # 技术分析
            'technical_analysis': [
                "技术发展呈现三大特点：1）基础架构升级，性能提升显著；2）应用场景丰富，落地案例增多；3）生态体系完善，协同效应增强。",
                "核心技术指标显著提升：计算效率提高3倍、能耗降低40%、部署成本降低50%。新一代产品已经进入测试阶段，预计明年实现商用。",
                "技术创新不断突破，专利申请数量年增长率超过100%。重点突破方向包括：架构优化、性能提升、安全增强等。"
            ],
            
            # 竞争分析
            'competition': [
                "市场竞争格局呈现出'一超多强'特征，龙头企业市占率超过30%，前五企业累计份额达到70%。中小企业主要在细分领域寻求突破。",
                "竞争焦点从纯技术实力转向综合解决方案能力。企业竞争策略更注重生态构建和服务升级，产品差异化趋势明显。",
                "竞争态势日益激烈，企业间兼并收购活跃。未来竞争将围绕技术创新、解决方案、服务能力等多个维度展开。"
            ],
            
            # 用户反馈
            'user_feedback': [
                "用户满意度调查显示：85%的企业对产品性能表示满意，78%认为投资回报符合预期，65%计划增加投入。主要改进建议集中在成本优化和服务响应速度。",
                "典型用户反馈：部署周期缩短50%，运营效率提升35%，维护成本降低40%。90%的用户愿意向其他企业推荐。",
                "用户反馈主要集中在：系统稳定性好、部署便捷、效果明显。但在成本控制和技术支持方面仍有提升空间。"
            ],
            
            # 投资数据
            'investment_data': [
                "2023年全球投资总额达1200亿美元，同比增长55%。其中，A轮占20%，B轮占30%，C轮及以后占50%。平均融资金额较去年提升40%。",
                "投资重点领域：基础设施（30%）、应用解决方案（40%）、技术服务（30%）。独角兽企业数量达到50家，总估值超过5000亿美元。",
                "资本市场表现活跃，IPO企业数量同比增长80%。二级市场表现良好，行业指数上涨65%，显著跑赢大盘。"
            ]
        }
        
        # 任务类型
        self.tasks = {
            'short_input': "生成详细内容",
            'long_input_long_output': "请进行深入分析并给出详细报告",
            'long_input_short_output': "请用一句话总结主要观点" 
        }

    def _get_random_content(self, content_type: str) -> str:
        """从内容库中随机获取内容"""
        if content_type in self.content_library:
            return np.random.choice(self.content_library[content_type])
        return f"[{content_type}]"

    def generate_short_input_long_output(self, count: int) -> List[Dict]:
        """生成短输入/长输出的提示词"""
        prompts = []
        for _ in range(count):
            topic = np.random.choice(self.content_library['topics'])
            template = np.random.choice(self.content_library['short_input'])
            prompt = template.format(topic=topic)
            
            tokens = len(self.tokenizer.encode(prompt))
            prompts.append({
                'prompt': prompt,
                'token_count': tokens,
                'type': 'short_input_long_output',
                'topic': topic
            })
        return prompts

    def generate_long_input_long_output(self, count: int) -> List[Dict]:
        """生成长输入/长输出的提示词"""
        prompts = []
        for _ in range(count):
            topic = np.random.choice(self.content_library['topics'])
            prompt = f"""
以下是一份关于{topic}的详细报告，{self.tasks['long_input_long_output']}：

背景介绍：
{self._get_random_content('background')}

主要内容：
{self._get_random_content('main_content')}

技术细节：
{self._get_random_content('technical_details')}

应用场景：
{self._get_random_content('applications')}

发展趋势：
{self._get_random_content('trends')}

问题与挑战：
{self._get_random_content('challenges')}
"""
            
            tokens = len(self.tokenizer.encode(prompt))
            prompts.append({
                'prompt': prompt,
                'token_count': tokens,
                'type': 'long_input_long_output',
                'topic': topic
            })
        return prompts

    def generate_long_input_short_output(self, count: int) -> List[Dict]:
        """生成长输入/短输出的提示词"""
        prompts = []
        for _ in range(count):
            topic = np.random.choice(self.content_library['topics'])
            prompt = f"""
请分析以下关于{topic}的详细资料，{self.tasks['long_input_short_output']}：

市场背景：
{self._get_random_content('market_background')}

技术分析：
{self._get_random_content('technical_analysis')}

竞争格局：
{self._get_random_content('competition')}

用户反馈：
{self._get_random_content('user_feedback')}

投资数据：
{self._get_random_content('investment_data')}
"""
            
            tokens = len(self.tokenizer.encode(prompt))
            prompts.append({
                'prompt': prompt,
                'token_count': tokens,
                'type': 'long_input_short_output',
                'topic': topic
            })
        return prompts
                
def main():
    print("欢迎使用LLM测试提示词生成工具")
    print("\n可选的生成模式：")
    print("1. 短输入/长输出 - 适用于生成任务")
    print("2. 长输入/长输出 - 适用于分析任务")
    print("3. 长输入/短输出 - 适用于分类任务")
    
    while True:
        try:
            mode = int(input("\n请选择生成模式 (1-3): "))
            if mode not in [1, 2, 3]:
                raise ValueError
            break
        except ValueError:
            print("请输入有效的选项 (1-3)")
    
    while True:
        try:
            count = int(input("请输入需要生成的提示词数量: "))
            if count <= 0:
                raise ValueError
            break
        except ValueError:
            print("请输入大于0的数字")
    
    generator = PromptGenerator()
    
    # 根据选择生成提示词
    if mode == 1:
        prompts = generator.generate_short_input_long_output(count)
        output_file = "./input/short_input_long_output_prompts.csv"
    elif mode == 2:
        prompts = generator.generate_long_input_long_output(count)
        output_file = "./input/long_input_long_output_prompts.csv"
    else:
        prompts = generator.generate_long_input_short_output(count)
        output_file = "./input/long_input_short_output_prompts.csv"
    
    # 保存结果
    df = pd.DataFrame(prompts)
    df.to_csv(output_file, index=False)
    
    # 打印统计信息
    print(f"\n生成完成！结果已保存至 {output_file}")
    print("\n统计信息：")
    print(f"总数量: {len(df)}")
    print(f"平均token数: {df['token_count'].mean():.2f}")
    print(f"最小token数: {df['token_count'].min()}")
    print(f"最大token数: {df['token_count'].max()}")
    
    # 打印样例
    print("\n示例提示词：")
    for i, row in df.head(2).iterrows():
        print(f"\n示例 {i+1} ({row['token_count']} tokens):")
        print(f"类型: {row['type']}")
        print(f"主题: {row['topic']}")
        print(f"提示词: {row['prompt'][:200]}...")

if __name__ == "__main__":
    main()
