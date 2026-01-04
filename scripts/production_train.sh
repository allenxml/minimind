#!/bin/bash
# 生产级训练脚本 - BF16混合精度 + 单GPU + 自动恢复
# 适用于 Linux/WSL2

set -e

# ============= 配置区域 =============
GPU_ID=0
DTYPE="bfloat16"
EPOCHS=50
BATCH_SIZE=32
ACCUMULATION_STEPS=8
LEARNING_RATE=5e-4
SAVE_INTERVAL=1200  # 约每30分钟（假设每步1.5秒: 30*60/1.5=1200）
LOG_INTERVAL=50
SAVE_DIR="./output"
MODEL_NAME="minimind_bf16"
HIDDEN_SIZE=512
NUM_LAYERS=8
MAX_SEQ_LEN=512
DATA_PATH="./dataset/pretrain_hq.jsonl"

# ============= 环境设置 =============
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTHONUNBUFFERED=1

# 创建输出目录
mkdir -p $SAVE_DIR
mkdir -p ./logs

# 生成日志文件名
LOG_FILE="./logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "     MiniMind 生产级训练"
echo "=========================================="
echo "时间: $(date)"
echo "GPU: $GPU_ID"
echo "精度: $DTYPE"
echo "Batch Size: $BATCH_SIZE x $ACCUMULATION_STEPS = $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "学习率: $LEARNING_RATE"
echo "保存间隔: 每 $SAVE_INTERVAL 步 (约 $((SAVE_INTERVAL * 15 / 600)) 分钟)"
echo "输出目录: $SAVE_DIR"
echo "日志文件: $LOG_FILE"
echo "=========================================="

# ============= 检查环境 =============
echo ""
echo "检查环境..."

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi
echo "✅ Python: $(python3 --version)"

# 检查 GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  nvidia-smi 不可用"
else
    echo ""
    echo "GPU 信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | head -n 1
fi

# 检查 PyTorch
python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" || {
    echo "❌ PyTorch 未安装"
    exit 1
}

# 检查 CUDA
python3 -c "import torch; print(f'✅ CUDA: {torch.cuda.is_available()}')" || echo "⚠️  CUDA 不可用"

# 检查数据文件
if [ ! -f "$DATA_PATH" ]; then
    echo "⚠️  数据文件不存在: $DATA_PATH"
    echo "   请确认数据路径是否正确"
fi

# ============= 开始训练 =============
echo ""
echo "=========================================="
echo "开始训练..."
echo "按 Ctrl+C 可以安全停止（会保存checkpoint）"
echo "=========================================="
echo ""

# 捕获中断信号
trap 'echo "收到停止信号，保存checkpoint..."; exit 0' INT TERM

# 开始训练
python3 trainer/train_pretrain.py \
    --device cuda:0 \
    --dtype $DTYPE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accumulation_steps $ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --grad_clip 1.0 \
    --save_interval $SAVE_INTERVAL \
    --log_interval $LOG_INTERVAL \
    --save_dir $SAVE_DIR \
    --save_weight $MODEL_NAME \
    --hidden_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_LAYERS \
    --max_seq_len $MAX_SEQ_LEN \
    --data_path $DATA_PATH \
    --from_resume 1 \
    --num_workers 4 \
    2>&1 | tee $LOG_FILE

# ============= 训练完成 =============
echo ""
echo "=========================================="
echo "     训练完成！"
echo "=========================================="
echo "完成时间: $(date)"
echo "模型保存位置: $SAVE_DIR"
echo "日志文件: $LOG_FILE"
echo ""

# 显示保存的模型
echo "已保存的模型:"
if ls $SAVE_DIR/*.pth 1> /dev/null 2>&1; then
    ls -lh $SAVE_DIR/*.pth
else
    echo "  无 .pth 文件"
fi

if ls $SAVE_DIR/*.safetensors 1> /dev/null 2>&1; then
    ls -lh $SAVE_DIR/*.safetensors
else
    echo "  无 .safetensors 文件"
fi

# 转换为 safetensors 格式
echo ""
if [ -f "$SAVE_DIR/${MODEL_NAME}_${HIDDEN_SIZE}.pth" ]; then
    echo "转换为 SafeTensors 格式..."
    if [ -f "scripts/convert_to_safetensors.py" ]; then
        python3 scripts/convert_to_safetensors.py "$SAVE_DIR/${MODEL_NAME}_${HIDDEN_SIZE}.pth" || echo "⚠️  转换失败"
    else
        echo "⚠️  转换脚本不存在"
    fi
fi

# 清理pip缓存（可选）
echo ""
read -p "是否清理pip缓存以释放空间? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "清理pip缓存..."
    pip cache purge
    echo "✅ 缓存已清理"
fi

echo ""
echo "=========================================="
echo "全部完成！"
echo "=========================================="

