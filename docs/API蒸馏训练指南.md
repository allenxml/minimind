# API 蒸馏训练指南

通过 OpenRouter API 从顶级大模型（GPT-5.2、Claude Opus 4.5、DeepSeek-R1 等）进行知识蒸馏，无需本地部署。

## 🌟 核心优势

- ✅ **无需高端显卡** - 通过 API 访问顶级模型
- ✅ **支持推理蒸馏** - 学习大模型的思维链
- ✅ **成本可控** - DeepSeek-R1 仅 $7/万样本
- ✅ **两种方式** - 用户提供问题 or 模型生成问答

---

## 🎯 两种数据生成方式

### 方式 1: 用户提供问题 → API 生成答案

**适合**: 有现成问题数据，想提升答案质量

```bash
python dataset/generate_distill_data_from_api.py \
    --api_key YOUR_OPENROUTER_KEY \
    --model anthropic/claude-opus-4.5 \
    --input_file dataset/sft_mini_512.jsonl \
    --output_file dataset/distill_opus45.jsonl \
    --mode reasoning
```

### 方式 2: API 自己生成问题+答案

**适合**: 从零开始构建特定领域数据集

```bash
python dataset/generate_qa_pairs_from_api.py \
    --api_key YOUR_OPENROUTER_KEY \
    --model anthropic/claude-sonnet-4.5 \
    --topic "Python编程基础和进阶" \
    --num_samples 1000 \
    --output_file dataset/qa_python.jsonl
```

---

## 🏆 推荐模型（基于 OpenRouter 2025 排名）

| 模型 | 成本($/万样本) | 适用场景 |
|------|---------------|---------|
| **Claude Opus 4.5** | $255 | 追求极致性能 ⭐⭐⭐⭐⭐ |
| **Claude Sonnet 4.5** | $51 | 性能与成本平衡 ⭐⭐⭐⭐⭐ |
| **DeepSeek-R1** | $7 | 最高性价比，支持推理 ⭐⭐⭐⭐⭐ |
| **Gemini 3 Flash** | $2 | 大规模低成本 ⭐⭐⭐⭐ |

---

## 🚀 快速开始

### 1. 获取 API Key

访问 [OpenRouter](https://openrouter.ai/) 注册并获取 API Key。

### 2. 测试连接

```bash
python examples/test_api_connection.py YOUR_OPENROUTER_KEY
```

### 3. 生成数据

**方式 1 - 改进已有数据**:
```bash
python dataset/generate_distill_data_from_api.py \
    --api_key YOUR_KEY \
    --model anthropic/claude-sonnet-4.5 \
    --input_file dataset/sft_mini_512.jsonl \
    --output_file dataset/enhanced.jsonl \
    --mode answer_only
```

**方式 2 - 从零生成**:
```bash
python dataset/generate_qa_pairs_from_api.py \
    --api_key YOUR_KEY \
    --model deepseek/deepseek-r1 \
    --topic "初中数学应用题" \
    --num_samples 500 \
    --output_file dataset/math_qa.jsonl \
    --mode reasoning
```

### 4. 训练模型

```bash
# 标准训练
python trainer/train_full_sft.py \
    --data_path dataset/enhanced.jsonl \
    --save_weight api_distill

# 推理蒸馏训练
python trainer/train_distill_reason.py \
    --data_path dataset/math_qa.jsonl \
    --save_weight reason_distill
```

---

## 💰 成本估算

以 **10,000 条数据**为例（每条 200 输入 + 300 输出 tokens）:

### 方式 1: 用户提供问题

| 模型 | 总成本 |
|------|--------|
| Claude Opus 4.5 | $255 |
| Claude Sonnet 4.5 | $51 |
| DeepSeek-R1 | $7 ⭐ |

### 方式 2: 模型生成问答

| 模型 | 总成本 |
|------|--------|
| Claude Opus 4.5 | $400 |
| Claude Sonnet 4.5 | $80 |
| DeepSeek-R1 | $12 ⭐ |

**成本优化建议**:
- 简单数据用低成本模型（Gemini 3 Flash）
- 核心数据用顶级模型（Claude Opus 4.5）
- 大量数据用性价比模型（DeepSeek-R1）

---

## 🎯 推荐策略

### 策略 1: 追求极致性能

```bash
# 使用顶级模型
python dataset/generate_distill_data_from_api.py \
    --model anthropic/claude-opus-4.5 \
    --mode reasoning
```

**成本**: ~$255/万样本  
**适合**: 预算充足，追求最佳效果

---

### 策略 2: 平衡性能成本

```bash
# 使用平衡模型
python dataset/generate_distill_data_from_api.py \
    --model anthropic/claude-sonnet-4.5 \
    --mode answer_only
```

**成本**: ~$51/万样本  
**适合**: 大多数用户（市场使用最广）

---

### 策略 3: 极致性价比

```bash
# 使用开源顶级模型
python dataset/generate_distill_data_from_api.py \
    --model deepseek/deepseek-r1 \
    --mode reasoning
```

**成本**: ~$7/万样本 ⭐  
**适合**: 预算有限，仍需高质量

---

### 策略 4: 混合策略（推荐）⭐⭐⭐

```bash
# 1. 核心数据用顶级模型（10%）
python dataset/generate_distill_data_from_api.py \
    --model anthropic/claude-opus-4.5 \
    --max_samples 1000

# 2. 扩展数据用性价比模型（90%）
python dataset/generate_distill_data_from_api.py \
    --model deepseek/deepseek-r1 \
    --max_samples 9000

# 3. 合并
cat data_core.jsonl data_extended.jsonl > data_final.jsonl
```

**成本**: ~$88/万样本（vs 全用 Opus: $2550）  
**节省**: 96.5%

---

## 📝 主要参数说明

### generate_distill_data_from_api.py（方式1）

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 教师模型 | `anthropic/claude-opus-4.5` |
| `--input_file` | 输入问题文件 | `dataset/questions.jsonl` |
| `--output_file` | 输出文件 | `dataset/enhanced.jsonl` |
| `--mode` | 输出模式 | `answer_only` / `reasoning` |
| `--max_tokens` | 最大生成长度 | `2048` |
| `--rate_limit_delay` | API 调用间隔(秒) | `1.0` |

### generate_qa_pairs_from_api.py（方式2）

| 参数 | 说明 | 示例 |
|------|------|------|
| `--topic` | 数据主题 | `"Python编程"` |
| `--num_samples` | 生成数量 | `1000` |
| `--difficulty` | 难度 | `easy`/`medium`/`hard` |
| `--language` | 语言 | `zh`/`en` |
| `--mode` | 输出模式 | `answer_only`/`reasoning` |

---

## ❓ 常见问题

### Q1: 两种方式如何选择？

**A:** 
- 有现成问题数据 → **方式1**
- 从零开始构建 → **方式2**  
- 推荐混合使用

### Q2: 推理模式(reasoning)是什么？

**A:** 推理模式让大模型输出详细的思考过程：

```json
{
  "answer": "<think>步骤1: ...\n步骤2: ...</think>\n<answer>最终答案</answer>"
}
```

适合数学、逻辑推理等需要思维链的任务。

### Q3: 如何降低成本？

**A:** 
1. 用 DeepSeek-R1（性价比最高）
2. 混合策略（少量顶级 + 大量性价比）
3. 控制 `max_tokens` 参数
4. 使用断点续传避免重复调用

### Q4: 生成速度很慢？

**A:** 
- 调整 `--rate_limit_delay`（默认1秒）
- 但太快可能触发限流，建议保持 0.5-1.0 秒

### Q5: 支持哪些主题？

**A:** 任意主题，例如：
- 编程: "Python/JavaScript/算法"
- 学科: "数学/物理/化学"
- 技能: "英语/写作/逻辑"

---

## 📚 相关资源

- 📖 [快速开始](./快速开始.md)
- 🔧 [训练指南](./训练指南.md)
- 💻 [推理部署](./推理部署.md)
- 🔗 [OpenRouter 排名](https://openrouter.ai/rankings)

---

## 🎉 总结

**核心流程**:

```
1. 获取 OpenRouter API Key
2. 选择教师模型（推荐 Claude Sonnet 4.5 或 DeepSeek-R1）
3. 选择生成方式（有数据用方式1，无数据用方式2）
4. 生成数据
5. 训练模型
```

**立即开始**:

```bash
# 测试 API
python examples/test_api_connection.py YOUR_KEY

# 查看帮助
python dataset/generate_distill_data_from_api.py --help
python dataset/generate_qa_pairs_from_api.py --help
```

祝训练顺利！🚀
