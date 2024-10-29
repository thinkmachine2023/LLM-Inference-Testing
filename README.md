# LLM-Inference-Testing
# LLM 推理服务性能测试工具 (LLM Inference Performance Testing Tools)

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


LLM推理服务的五个核心性能指标计算方法和详细说明：

#### 1. 首个Token响应时间 (TTFT - Time To First Token)

```python
ttft = (first_token_time - start_time) * 1000  # 转换为毫秒
```

**说明：**
- **定义**：从发送请求到接收到第一个生成token的时间
- **单位**：毫秒 (ms)
- **计算要点**：
  - `start_time`：请求发送时间戳
  - `first_token_time`：首个token接收时间戳
  - 需要开启stream模式才能准确计算
- **性能标准**：
  - 优秀：< 200ms
  - 良好：200-500ms
  - 需优化：> 500ms
- **影响因素**：
  - 网络延迟
  - 推理服务启动时间
  - 输入处理时间
  - 模型加载状态

#### 2. 单Token生成时间 (TPOT - Time Per Output Token)

```python
generation_time = end_time - first_token_time
tpot = (generation_time / output_tokens) * 1000  # 毫秒/token
```

**说明：**
- **定义**：生成每个token的平均时间
- **单位**：毫秒/token (ms/token)
- **计算要点**：
  - `generation_time`：总生成时间（不包含首个token的等待时间）
  - `output_tokens`：生成的token总数
  - 需排除TTFT的影响
- **性能标准**：
  - 优秀：< 30ms/token
  - 良好：30-50ms/token
  - 需优化：> 50ms/token
- **影响因素**：
  - 模型计算能力
  - 硬件配置
  - batch大小
  - 上下文长度

#### 3. 端到端延迟 (Latency)

```python
latency = (end_time - start_time) * 1000  # 转换为毫秒
# 或者
latency = ttft + (tpot * output_tokens)  # 理论计算
```

**说明：**
- **定义**：完整请求的总响应时间
- **单位**：毫秒 (ms)
- **计算要点**：
  - 包含完整的请求生命周期
  - 从发送请求到接收全部响应
  - 考虑所有处理环节
- **性能标准**：
  - 因输出长度不同而异
  - 短文本（<100 tokens）：建议 < 2000ms
  - 长文本：可接受延迟 = 基础延迟(TTFT) + 单token时间 × token数量
- **影响因素**：
  - TTFT
  - TPOT
  - 输出长度
  - 网络状况

#### 4. Token吞吐量 (TPS - Tokens Per Second)

```python
tps = output_tokens / total_time  # total_time单位为秒
```

**说明：**
- **定义**：每秒生成的token数量
- **单位**：tokens/second
- **计算要点**：
  - `output_tokens`：生成的token总数
  - `total_time`：总处理时间（秒）
  - 批处理场景下需考虑并发
- **性能标准**：
  - 单请求：
    - 优秀：> 70 tokens/s
    - 良好：30-70 tokens/s
    - 需优化：< 30 tokens/s
  - 批处理场景下会随batch size变化
- **影响因素**：
  - 模型性能
  - 硬件配置
  - 批处理大小
  - 系统负载

#### 5. 请求吞吐量 (RPS - Requests Per Second)

```python
rps = 1 / total_time  # 单请求
# 或
rps = request_count / total_test_time  # 批处理场景
```

**说明：**
- **定义**：每秒处理的请求数量
- **单位**：requests/second
- **计算要点**：
  - 单请求：请求处理时间的倒数
  - 批处理：总请求数/总测试时间
  - 需考虑并发和排队延迟
- **性能标准**：
  - 因请求复杂度不同而异
  - 简单请求：> 1 RPS
  - 复杂请求：0.1-1 RPS
  - 批处理：随batch size优化
- **影响因素**：
  - 系统资源
  - 并发能力
  - 请求复杂度
  - 负载均衡

### 指标关系

1. **计算关系**
```
总延迟(Latency) = TTFT + 生成时间
生成时间 = TPOT × 输出token数
理论最大RPS = 1 / Latency
实际RPS ≤ 理论最大RPS
```

2. **优化建议**
- TTFT优化：
  - 模型预热
  - 资源预分配
  - 网络优化

- TPOT优化：
  - 硬件升级
  - 模型量化
  - 批处理优化

- 吞吐量优化：
  - 合理的batch size
  - 请求排队策略
  - 负载均衡
  - 资源弹性扩展


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
