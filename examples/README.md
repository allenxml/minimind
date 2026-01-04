# 示例脚本

本目录包含各种示例脚本，帮助你快速上手 MiniMind 的各项功能。

## API 蒸馏训练示例

### 📌 两种数据生成方式

在开始之前，了解两种不同的数据生成方式：

| 方式 | 脚本 | 适用场景 |
|------|------|---------|
| **方式1** | `generate_distill_data_from_api.py` | ✅ 有现成问题数据 |
| **方式2** | `generate_qa_pairs_from_api.py` | ✅ 从零开始构建数据 |

详见 [数据生成方式对比](../docs/数据生成方式对比.md)

---

### 1. 测试 API 连接

在开始之前，先测试你的 OpenRouter API Key 是否有效:

```bash
python test_api_connection.py YOUR_OPENROUTER_KEY
```

这个脚本会:
- ✅ 验证 API Key 是否有效
- 📋 列出可用的模型和价格
- 🧪 执行一次简单的 API 调用测试

### 2. 完整蒸馏训练示例

运行完整的 API 蒸馏训练流程（生成数据 + 训练 + 测试）:

```bash
# 设置环境变量
export OPENROUTER_API_KEY="your_key_here"

# 运行示例脚本
bash api_distillation_example.sh
```

这个脚本会:
1. 从 Claude Sonnet 4.5 生成 50 条推理蒸馏数据（演示用）
2. 使用生成的数据训练模型
3. 测试训练好的模型

**注意**: 示例脚本仅生成 50 条数据用于演示。实际训练建议使用更多数据。

---

### 3. 生成问答数据（方式2）

如果你没有现成的问题数据，可以让模型自己生成：

```bash
# 生成 Python 编程问答数据
python ../dataset/generate_qa_pairs_from_api.py \
    --api_key YOUR_OPENROUTER_KEY \
    --model anthropic/claude-sonnet-4.5 \
    --topic "Python编程基础和进阶" \
    --num_samples 100 \
    --output_file ../dataset/qa_python_demo.jsonl \
    --difficulty medium \
    --language zh

# 训练
python ../trainer/train_full_sft.py \
    --data_path ../dataset/qa_python_demo.jsonl \
    --save_weight qa_python
```

**主题示例:**
- 编程: "Python编程"、"JavaScript前端"、"算法与数据结构"
- 学科: "高中数学"、"初中物理"
- 技能: "英语语法"、"写作技巧"、"逻辑推理"

---

## 其他示例

更多示例和详细文档请参考:
- [API蒸馏训练指南](../docs/API蒸馏训练指南.md) - 完整的使用指南（包含两种方式对比）
- [快速开始](../docs/快速开始.md) - 基础入门教程
- [训练指南](../docs/训练指南.md) - 详细的训练说明

