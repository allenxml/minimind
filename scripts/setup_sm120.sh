#!/bin/bash
# MiniMind GPU环境配置脚本 - 支持Blackwell sm_120
# 适用于Linux/Mac系统

set -e

echo "============================================================"
echo "   MiniMind 环境配置工具"
echo "   支持 Blackwell 架构 (sm_120)"
echo "============================================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到Python3，请先安装Python 3.8+"
    exit 1
fi

echo "[1/4] 检测到Python版本:"
python3 --version
echo ""

# 升级pip
echo "[2/5] 升级pip到最新版本..."
python3 -m pip install --upgrade pip
echo ""

# 选择pip源
echo "[3/5] 选择pip镜像源..."
echo ""
echo "请选择pip镜像源:"
echo "  1. 清华源 (推荐-国内)"
echo "  2. 阿里源 (国内)"
echo "  3. 官方源 (国外)"
echo ""
read -p "请输入选项 (1-3): " mirror_choice

case $mirror_choice in
    1)
        echo "配置清华源..."
        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
        PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
        ;;
    2)
        echo "配置阿里源..."
        pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
        PIP_INDEX="https://mirrors.aliyun.com/pypi/simple/"
        ;;
    *)
        echo "使用官方源"
        PIP_INDEX=""
        ;;
esac
echo ""

# 安装PyTorch (支持sm_120)
echo "[4/5] 安装PyTorch (支持Blackwell架构)..."
echo ""
echo "请选择CUDA版本:"
echo "  1. CUDA 12.4 (推荐)"
echo "  2. CUDA 12.6"
echo "  3. CUDA 11.8 (老显卡)"
echo "  4. CPU版本"
echo "  5. 跳过PyTorch安装"
echo ""
read -p "请输入选项 (1-5): " cuda_choice

if [ "$mirror_choice" = "1" ]; then
    # 清华源PyTorch镜像
    case $cuda_choice in
        1)
            echo "正在从清华源安装 PyTorch with CUDA 12.4..."
            echo "提示: 显示详细进度，预计3-5分钟"
            pip install torch>=2.7.0 torchvision>=0.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple -v
            ;;
        2)
            echo "正在从清华源安装 PyTorch with CUDA 12.6..."
            echo "提示: 显示详细进度，预计3-5分钟"
            pip install torch>=2.7.0 torchvision>=0.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple -v
            ;;
        3)
            echo "正在从清华源安装 PyTorch with CUDA 11.8..."
            echo "提示: 显示详细进度，预计3-5分钟"
            pip install torch>=2.7.0 torchvision>=0.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple -v
            ;;
        4)
            echo "正在从清华源安装 PyTorch CPU版本..."
            echo "提示: 显示详细进度，预计3-5分钟"
            pip install torch>=2.7.0 torchvision>=0.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple -v
            ;;
        *)
            echo "跳过PyTorch安装"
            ;;
    esac
else
    # 官方源PyTorch
    case $cuda_choice in
        1)
            echo "正在安装 PyTorch with CUDA 12.4..."
            echo "提示: 显示详细进度，预计时间较长"
            pip install torch>=2.7.0 torchvision>=0.22.0 --index-url https://download.pytorch.org/whl/cu124 -v
            ;;
        2)
            echo "正在安装 PyTorch with CUDA 12.6..."
            echo "提示: 显示详细进度，预计时间较长"
            pip install torch>=2.7.0 torchvision>=0.22.0 --index-url https://download.pytorch.org/whl/cu126 -v
            ;;
        3)
            echo "正在安装 PyTorch with CUDA 11.8..."
            echo "提示: 显示详细进度，预计时间较长"
            pip install torch>=2.7.0 torchvision>=0.22.0 --index-url https://download.pytorch.org/whl/cu118 -v
            ;;
        4)
            echo "正在安装 PyTorch CPU版本..."
            echo "提示: 显示详细进度，预计时间较长"
            pip install torch>=2.7.0 torchvision>=0.22.0 --index-url https://download.pytorch.org/whl/cpu -v
            ;;
        *)
            echo "跳过PyTorch安装"
            ;;
    esac
fi
echo ""

# 安装其他依赖
echo "[5/5] 安装其他依赖包..."
echo "提示: 显示详细进度，预计1-2分钟"
if [ -n "$PIP_INDEX" ]; then
    pip install -r requirements.txt -i $PIP_INDEX -v
else
    pip install -r requirements.txt -v
fi
echo ""

# 运行GPU检测
echo "============================================================"
echo "   环境配置完成，正在检测GPU..."
echo "============================================================"
echo ""
python3 scripts/check_gpu_compatibility.py

echo ""
echo "============================================================"
echo "   配置完成！"
echo "============================================================"
echo ""
echo "下一步:"
echo "  - 运行训练: python3 trainer/train_pretrain.py --epochs 1 --batch_size 32"
echo "  - 查看文档: GPU_COMPATIBILITY.md"
echo ""

