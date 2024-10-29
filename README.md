# LLM-Inference-Testing
# LLM 推理服务测试工具 (LLM Inference Performance Testing Tools)

用于评估和测试大语言模型推理服务端到端性能的专业工具集，包含负载生成和性能测试功能。

## 工具概览

### 1. 负载生成器 (Prompt Generator)
生成不同输入输出Token分布模式的测试用例：
- 短输入/长输出模式：适用于内容生成场景
- 长输入/长输出模式：适用于深度分析场景
- 长输入/短输出模式：适用于分类、摘要或信息提取场景

### 2. 性能测试器 (Performance Tester)
提供全面的推理服务性能评估指标：
- 首个Token响应时间 (TTFT - Time To First Token)
- 单Token生成时间 (TPOT - Time Per Output Token)
- 端到端延迟 (End-to-End Latency)
- Token吞吐量 (TPS - Tokens Per Second)
- 请求吞吐量 (RPS - Requests Per Second)

## 环境要求

- Python 3.7+
- 依赖包：
  ```
  pandas
  numpy
  modelscope
  openai
  tqdm
  tiktoken
  ```

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/thinkmachine2023/llm-inference-testing.git
cd llm-inference-testing
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用指南

### 1. 负载生成

```bash
python prompt_generator.py
```

交互式配置：
- 选择生成模式 (1-3)
- 指定生成数量

输出文件位于 `input/` 目录：
- `short_input_long_output_prompts.csv`
- `long_input_long_output_prompts.csv`
- `long_input_short_output_prompts.csv`

### 2. 性能测试

```bash
python performance_test.py
```

配置参数：
- API配置（密钥、基础URL、模型名称）
- 批处理大小（逗号分隔，如 "1,2,4,8"）

输出文件：
- `output_performance_metrics_batch{size}.csv`：各批次详细指标
- `batch_size_comparison.csv`：批次间对比分析

## 性能指标说明

### 核心指标：
1. **首个Token响应时间 (TTFT)**
   - 衡量推理服务的初始响应速度
   - 对实时交互场景至关重要

2. **单Token生成时间 (TPOT)**
   - 评估Token生成效率
   - 反映模型推理性能

3. **延迟 (Latency)**
   - 完整请求的响应时间
   - 计算公式：TTFT + (TPOT × token数量)

4. **吞吐量 (Throughput)**
   - TPS：每秒处理的Token数
   - RPS：每秒处理的请求数

### 性能数据格式

测试结果CSV格式：
```csv
prompt,input_tokens,output_tokens,ttft,tpot,latency,tps,rps,status
"测试文本...",45,128,181.55,42.33,8899.57,76.39,0.11,success
```

## 项目结构

```
llm-inference-testing/
├── prompt_generator.py    # 负载生成工具
├── performance_test.py    # 性能测试工具
├── requirements.txt      # 项目依赖
├── README.md            # 说明文档
├── LICENSE             # 许可证
├── input/              # 测试用例
│   ├── short_input_long_output_prompts.csv
│   ├── long_input_long_output_prompts.csv
│   └── long_input_short_output_prompts.csv
└── output/             # 测试结果
    ├── output_performance_metrics_batch1.csv
    ├── output_performance_metrics_batch2.csv
    └── batch_size_comparison.csv
```

## 功能特性

### 负载生成器
- 专业领域内容模板
- Token数量计算
- 统计分析
- 多种输出模式

### 性能测试器
- 批处理支持
- 异步请求处理
- 全面性能指标
- 对比分析
- 进度监控

## 测试结果示例

### Token分布
```
模式                    输入Token数   输出Token数
短输入/长输出           30-50        200+
长输入/长输出           500-800      300+
长输入/短输出           500-800      10-30
```

### 性能指标示例
```
批次大小  平均TTFT(ms)  平均延迟(ms)   TPS    RPS
1        181.55       8899.57       76.39  0.11
2        256.78       9856.78       65.45  0.19
4        389.67       12456.89      45.67  0.35
8        567.89       15678.90      35.78  0.62
```

## 最佳实践

1. **负载生成**
   - 先生成小批量样本验证
   - 验证Token数量符合预期
   - 使用多样化主题测试

2. **性能测试**
   - 从小批量开始测试
   - 监控系统资源使用
   - 注意API限流
   - 确保充足的测试时间
   - 收集多组数据取平均值

3. **结果分析**
   - 关注异常值和波动
   - 对比不同批次的性能曲线
   - 评估系统稳定性
   - 识别性能瓶颈

## 故障排除

1. **常见问题**
   - API连接超时：检查网络和配置
   - Token计数不准：更新tokenizer
   - 性能波动大：增加样本量和测试时间

2. **错误处理**
   - 详细的错误日志
   - 请求重试机制
   - 异常状态恢复

## 贡献指南

欢迎提交Pull Request进行贡献！

## 开源许可

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件

## 作者

- ThinkMachine -
  
## 致谢



## 联系方式

- 项目地址: [https://github.com/thinkmachine2023/llm-inference-testing](https://github.com/thinkmachine2023/llm-inference-testing)
