#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Performance Test
~~~~~~~~~~~~~~~~~~~~~~~~
Author: Liang Zhun
Version: 1.0.0
Created: October 2024
License: Apache License 2.0
"""

import asyncio
import csv
import time
from typing import List, Dict
import aiohttp
from openai import AsyncOpenAI
import pandas as pd
from tqdm import tqdm
import tiktoken

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'input_tokens': 0,
            'output_tokens': 0,
            'ttft': 0,        # Time To First Token
            'tpot': 0,        # Time Per Output Token
            'latency': 0,     # Overall Latency
            'tps': 0,         # Tokens Per Second
            'rps': 0          # Requests Per Second
        }
    
    def calculate_metrics(self, 
                         start_time: float,
                         first_token_time: float,
                         end_time: float,
                         input_tokens: int,
                         output_tokens: int) -> Dict:
        """计算所有性能指标"""
        # 基础时间计算
        total_time = end_time - start_time
        generation_time = end_time - first_token_time
        
        # 1. TTFT (Time To First Token)
        ttft = (first_token_time - start_time) * 1000  # 转换为毫秒
        
        # 2. TPOT (Time Per Output Token)
        tpot = (generation_time / output_tokens) * 1000 if output_tokens > 0 else 0
        
        # 3. Latency (Total Response Time)
        latency = total_time * 1000  # 转换为毫秒
        
        # 4. TPS (Tokens Per Second)
        tps = output_tokens / total_time if total_time > 0 else 0
        
        # 5. RPS (Requests Per Second)
        rps = 1 / total_time if total_time > 0 else 0
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'ttft': ttft,
            'tpot': tpot,
            'latency': latency,
            'tps': tps,
            'rps': rps
        }


class BatchProcessor:
    def __init__(self, api_key: str, base_url: str = None, model: str = "gpt-3.5-turbo", batch_size: int = 5):
        # 配置OpenAI客户端
        client_params = {
            "api_key": api_key,
        }
        if base_url:
            client_params["base_url"] = base_url
            
        self.client = AsyncOpenAI(**client_params)
        self.model = model
        self.batch_size = batch_size
        self.base_url = base_url
        
        # 初始化tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print(f"Warning: No specific tokenizer found for {model}, using cl100k_base instead")
            self.encoding = tiktoken.get_encoding("cl100k_base")
            
        self.performance_monitor = PerformanceMonitor()
        
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    async def process_single_request(self, prompt: str) -> Dict:
        start_time = time.time()
        first_token_time = None
        
        try:
            input_tokens = self.count_tokens(prompt)
            
            # 创建聊天完成请求
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            
            full_response = ""
            async for chunk in response:
                if first_token_time is None:
                    first_token_time = time.time()
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
            
            end_time = time.time()
            output_tokens = self.count_tokens(full_response)
            
            # 计算所有性能指标
            metrics = self.performance_monitor.calculate_metrics(
                start_time=start_time,
                first_token_time=first_token_time,
                end_time=end_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

            return {
                "prompt": prompt,
                "response": full_response,
                **metrics,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "prompt": prompt,
                "error": str(e),
                "status": "failed"
            }

    async def process_batch(self, prompts: List[str]) -> List[Dict]:
        tasks = [self.process_single_request(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def process_all(self, input_file: str, output_file: str):
        df = pd.read_csv(input_file)
        prompts = df['prompt'].tolist()
        
        results = []
        with tqdm(total=len(prompts)) as pbar:
            for i in range(0, len(prompts), self.batch_size):
                batch = prompts[i:i + self.batch_size]
                batch_results = await self.process_batch(batch)
                results.extend(batch_results)
                pbar.update(len(batch))

        successful_results = [r for r in results if r["status"] == "success"]
        if successful_results:
            df_results = pd.DataFrame(successful_results)
            
            # 计算性能统计
            stats = df_results.agg({
                'input_tokens': ['mean', 'min', 'max'],
                'output_tokens': ['mean', 'min', 'max'],
                'ttft': ['mean', 'min', 'max'],
                'tpot': ['mean', 'min', 'max'],
                'latency': ['mean', 'min', 'max'],
                'tps': ['mean', 'min', 'max'],
                'rps': ['mean', 'min', 'max']
            }).round(2)
            
            # 添加环境信息
            env_info = {
                'model': self.model,
                'batch_size': self.batch_size,
                'base_url': self.base_url or 'default',
                'total_requests': len(df_results),
                'success_rate': f"{(len(successful_results) / len(prompts)) * 100:.2f}%"
            }
            
            print("\nEnvironment Information:")
            for key, value in env_info.items():
                print(f"{key}: {value}")
            
            print("\nPerformance Statistics:")
            print(stats)
            
            # 保存详细结果
            df_results.to_csv(output_file, index=False)
            
            # 保存统计结果，包含环境信息
            stats_df = pd.DataFrame(stats)
            stats_df.loc['environment'] = pd.Series(env_info)
            stats_df.to_csv(output_file.replace('.csv', '_stats.csv'))
        
        failed_results = [r for r in results if r["status"] == "failed"]
        if failed_results:
            print("\nFailed requests:")
            for result in failed_results:
                print(f"Prompt: {result['prompt']}")
                print(f"Error: {result['error']}\n")

async def run_batch_test(
    api_key: str,
    base_url: str,
    input_file: str,
    batch_size: int,
    model: str
) -> str:
    """运行单个batch size的测试"""
    output_file = f"./output/output_performance_metrics_batch{batch_size}.csv"
    
    print(f"\n开始测试 batch_size = {batch_size}")
    processor = BatchProcessor(
        api_key=api_key,
        base_url=base_url,
        model=model,
        batch_size=batch_size
    )
    
    await processor.process_all(input_file, output_file)
    return output_file

async def run_comparative_analysis(output_files: List[str]):
    """对不同batch size的结果进行对比分析"""
    print("\n开始生成对比分析报告...")
    
    # 收集所有batch size的统计数据
    comparative_stats = []
    for file in output_files:
        batch_size = int(file.split('batch')[-1].split('.')[0])
        df = pd.read_csv(file)
        
        stats = df.agg({
            'input_tokens': ['mean', 'std', 'min', 'max'],
            'output_tokens': ['mean', 'std', 'min', 'max'],
            'ttft': ['mean', 'std', 'min', 'max'],
            'tpot': ['mean', 'std', 'min', 'max'],
            'latency': ['mean', 'std', 'min', 'max'],
            'tps': ['mean', 'std', 'min', 'max'],
            'rps': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        stats_dict = {
            'batch_size': batch_size,
            'sample_size': len(df),
            **{f"{metric}_{stat}": value 
               for metric, values in stats.items() 
               for stat, value in zip(['mean', 'std', 'min', 'max'], values)}
        }
        comparative_stats.append(stats_dict)
    
    # 创建对比分析DataFrame
    comparative_df = pd.DataFrame(comparative_stats)
    comparative_df.sort_values('batch_size', inplace=True)
    
    # 保存对比分析结果
    comparison_file = './output/batch_size_comparison.csv'
    comparative_df.to_csv(comparison_file, index=False)
    
    # 打印关键指标对比
    print("\n不同batch size的关键性能指标对比：")
    print("\n1. 平均延迟 (ms):")
    print(comparative_df[['batch_size', 'latency_mean', 'latency_std']].to_string(index=False))
    
    print("\n2. TPS (Tokens Per Second):")
    print(comparative_df[['batch_size', 'tps_mean', 'tps_std']].to_string(index=False))
    
    print("\n3. TTFT (Time To First Token) (ms):")
    print(comparative_df[['batch_size', 'ttft_mean', 'ttft_std']].to_string(index=False))
    
    return comparison_file

async def main():
    # 获取用户输入
    print("欢迎使用批量性能测试工具")
    print("请输入测试配置：")
    
    # 获取API配置
    # api_key = input("请输入API Key: ").strip()
    # base_url = input("请输入base URL (直接回车使用默认值): ").strip()
    # model = input("请输入模型名称 (直接回车使用gpt-3.5-turbo): ").strip() or "gpt-3.5-turbo"

    api_key = "sk-123456"
    base_url = "http://192.168.10.250:8002/v1"  # 添加自定义endpoint
    model = "/opt/llm/Qwen/Qwen2-72B-Instruct-GPTQ-Int3"
    
    # 获取batch sizes
    while True:
        batch_sizes_input = input("请输入要测试的batch sizes (用逗号分隔，例如: 1,2,4,8): ").strip()
        try:
            batch_sizes = [int(size.strip()) for size in batch_sizes_input.split(',')]
            if all(size > 0 for size in batch_sizes):
                break
            else:
                print("错误：batch size必须大于0")
        except ValueError:
            print("错误：请输入有效的数字，用逗号分隔")
    
    input_file = "./input/short_input_long_output_prompts.csv"
    
    # 确认开始测试
    print("\n测试配置：")
    print(f"Model: {model}")
    print(f"Base URL: {base_url or '默认'}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Input file: {input_file}")
    
    confirm = input("\n是否开始测试? (y/n): ").strip().lower()
    if confirm != 'y':
        print("测试已取消")
        return
    
    # 执行所有batch size的测试
    output_files = []
    try:
        for batch_size in batch_sizes:
            output_file = await run_batch_test(
                api_key=api_key,
                base_url=base_url,
                input_file=input_file,
                batch_size=batch_size,
                model=model
            )
            output_files.append(output_file)
        
        # 生成对比分析报告
        comparison_file = await run_comparative_analysis(output_files)
        
        print(f"\n测试完成！")
        print(f"各批次详细结果已保存至: {', '.join(output_files)}")
        print(f"对比分析报告已保存至: {comparison_file}")
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n测试已被用户中断")
    except Exception as e:
        print(f"\n程序运行错误: {str(e)}")
        import traceback
        traceback.print_exc()