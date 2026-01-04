@echo off
REM MiniMind GPU环境配置脚本 - 支持Blackwell sm_120
REM 适用于Windows系统

echo ============================================================
echo    MiniMind 环境配置工具
echo    支持 Blackwell 架构 (sm_120)
echo ============================================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo [1/4] 检测到Python版本:
python --version
echo.

REM 升级pip
echo [2/5] 升级pip到最新版本...
python -m pip install --upgrade pip
echo.

REM 选择pip源
echo [3/5] 选择pip镜像源...
echo.
echo 请选择pip镜像源:
echo   1. 清华源 (推荐-国内)
echo   2. 阿里源 (国内)
echo   3. 官方源 (国外)
echo.
set /p mirror_choice="请输入选项 (1-3): "

if "%mirror_choice%"=="1" (
    echo 配置清华源...
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
    set PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
) else if "%mirror_choice%"=="2" (
    echo 配置阿里源...
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
    set PIP_INDEX=https://mirrors.aliyun.com/pypi/simple/
) else (
    echo 使用官方源
    set PIP_INDEX=
)
echo.

REM 安装PyTorch (支持sm_120)
echo [4/5] 安装PyTorch (支持Blackwell架构)...
echo.
echo 请选择CUDA版本:
echo   1. CUDA 12.4 (推荐)
echo   2. CUDA 12.6
echo   3. CUDA 11.8 (老显卡)
echo   4. 跳过PyTorch安装
echo.
set /p cuda_choice="请输入选项 (1-4): "

if "%mirror_choice%"=="1" (
    REM 清华源PyTorch镜像
    if "%cuda_choice%"=="1" (
        echo 正在从清华源安装 PyTorch with CUDA 12.4...
        echo 提示: 显示详细进度，预计3-5分钟
        pip install torch>=2.7.0 torchvision>=0.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple -v
    ) else if "%cuda_choice%"=="2" (
        echo 正在从清华源安装 PyTorch with CUDA 12.6...
        echo 提示: 显示详细进度，预计3-5分钟
        pip install torch>=2.7.0 torchvision>=0.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple -v
    ) else if "%cuda_choice%"=="3" (
        echo 正在从清华源安装 PyTorch with CUDA 11.8...
        echo 提示: 显示详细进度，预计3-5分钟
        pip install torch>=2.7.0 torchvision>=0.22.0 -i https://pypi.tuna.tsinghua.edu.cn/simple -v
    ) else (
        echo 跳过PyTorch安装
    )
) else (
    REM 官方源PyTorch
    if "%cuda_choice%"=="1" (
        echo 正在安装 PyTorch with CUDA 12.4...
        echo 提示: 显示详细进度，预计时间较长
        pip install torch>=2.7.0 torchvision>=0.22.0 --index-url https://download.pytorch.org/whl/cu124 -v
    ) else if "%cuda_choice%"=="2" (
        echo 正在安装 PyTorch with CUDA 12.6...
        echo 提示: 显示详细进度，预计时间较长
        pip install torch>=2.7.0 torchvision>=0.22.0 --index-url https://download.pytorch.org/whl/cu126 -v
    ) else if "%cuda_choice%"=="3" (
        echo 正在安装 PyTorch with CUDA 11.8...
        echo 提示: 显示详细进度，预计时间较长
        pip install torch>=2.7.0 torchvision>=0.22.0 --index-url https://download.pytorch.org/whl/cu118 -v
    ) else (
        echo 跳过PyTorch安装
    )
)
echo.

REM 安装其他依赖
echo [5/5] 安装其他依赖包...
echo 提示: 显示详细进度，预计1-2分钟
if defined PIP_INDEX (
    pip install -r requirements.txt -i %PIP_INDEX% -v
) else (
    pip install -r requirements.txt -v
)
echo.

REM 运行GPU检测
echo ============================================================
echo    环境配置完成，正在检测GPU...
echo ============================================================
echo.
python scripts/check_gpu_compatibility.py

echo.
echo ============================================================
echo    配置完成！
echo ============================================================
echo.
echo 下一步:
echo   - 运行训练: python trainer/train_pretrain.py --epochs 1 --batch_size 32
echo   - 查看文档: GPU_COMPATIBILITY.md
echo.
pause

