#!/bin/bash

# API 蒸馏训练完整示例脚本
# 
# 本脚本演示如何使用 OpenRouter API 从大模型进行知识蒸馏
# 
# 使用前请设置环境变量:
# export OPENROUTER_API_KEY="your_api_key_here"

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  MiniMind API 蒸馏训练示例${NC}"
echo -e "${BLUE}======================================${NC}"

# 检查 API Key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo -e "${YELLOW}警告: 未设置 OPENROUTER_API_KEY 环境变量${NC}"
    echo "请运行: export OPENROUTER_API_KEY='your_key'"
    exit 1
fi

# 配置参数
INPUT_DATA="../dataset/sft_mini_512.jsonl"
OUTPUT_DATA="../dataset/distill_api_demo.jsonl"

# 教师模型选择（可根据需求修改）:
# - anthropic/claude-opus-4.5: 顶级性能，成本较高
# - anthropic/claude-sonnet-4.5: 性能优秀，性价比高
# - deepseek/deepseek-r1: 最高性价比
# - google/gemini-3-flash-preview: 超低成本
TEACHER_MODEL="anthropic/claude-sonnet-4.5"  # 推荐：性能与成本平衡

MODE="reasoning"
MAX_SAMPLES=50  # 演示用，仅生成50条

echo -e "\n${GREEN}[1/3] 生成蒸馏数据${NC}"
echo "教师模型: $TEACHER_MODEL"
echo "输入文件: $INPUT_DATA"
echo "输出文件: $OUTPUT_DATA"
echo "模式: $MODE"
echo "样本数: $MAX_SAMPLES (演示用)"

python ../dataset/generate_distill_data_from_api.py \
    --api_key "$OPENROUTER_API_KEY" \
    --model "$TEACHER_MODEL" \
    --input_file "$INPUT_DATA" \
    --output_file "$OUTPUT_DATA" \
    --mode "$MODE" \
    --reasoning_effort high \
    --temperature 0.8 \
    --max_tokens 2048 \
    --max_samples $MAX_SAMPLES \
    --rate_limit_delay 1.0

echo -e "\n${GREEN}[2/3] 开始推理蒸馏训练${NC}"

python ../trainer/train_distill_reason.py \
    --data_path "$OUTPUT_DATA" \
    --save_weight api_distill_demo \
    --from_weight full_sft \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --epochs 1 \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --max_seq_len 1024 \
    --log_interval 10

echo -e "\n${GREEN}[3/3] 测试训练好的模型${NC}"

python ../scripts/infer_chat.py \
    --model_name ../out/api_distill_demo_512.pth \
    --prompt "小明有5个苹果，吃了2个，又买了3个，现在有几个？请详细说明计算过程。"

echo -e "\n${GREEN}✅ 完成! 模型已保存到 ../out/api_distill_demo_512.pth${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "下一步建议:"
echo -e "1. 使用更多数据重新训练 (移除 --max_samples 参数)"
echo -e "2. 尝试不同的教师模型 (如 openai/gpt-4-turbo)"
echo -e "3. 查看完整文档: docs/API蒸馏训练指南.md"
echo -e "${BLUE}======================================${NC}"

