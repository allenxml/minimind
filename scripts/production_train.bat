@echo off
REM 生产级训练脚本 - BF16混合精度 + 单GPU + 自动恢复
REM 适用于 Windows

setlocal enabledelayedexpansion

REM ============= 配置区域 =============
set GPU_ID=0
set DTYPE=bfloat16
set EPOCHS=50
set BATCH_SIZE=32
set ACCUMULATION_STEPS=8
set LEARNING_RATE=5e-4
set SAVE_INTERVAL=1200
set LOG_INTERVAL=50
set SAVE_DIR=.\output
set MODEL_NAME=minimind_bf16
set HIDDEN_SIZE=512
set NUM_LAYERS=8
set MAX_SEQ_LEN=512
set DATA_PATH=.\dataset\pretrain_hq.jsonl

REM ============= 环境设置 =============
set CUDA_VISIBLE_DEVICES=%GPU_ID%
set PYTHONUNBUFFERED=1

REM 创建输出目录
if not exist %SAVE_DIR% mkdir %SAVE_DIR%
if not exist .\logs mkdir .\logs

REM 生成日志文件名
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/:" %%a in ("%TIME%") do (set mytime=%%a%%b)
set LOG_FILE=.\logs\train_%mydate%_%mytime%.log

echo ==========================================
echo      MiniMind 生产级训练
echo ==========================================
echo 时间: %date% %time%
echo GPU: %GPU_ID%
echo 精度: %DTYPE%
echo Batch Size: %BATCH_SIZE% x %ACCUMULATION_STEPS%
echo 保存间隔: 每 %SAVE_INTERVAL% 步
echo 输出目录: %SAVE_DIR%
echo 日志文件: %LOG_FILE%
echo ==========================================
echo.

REM ============= 检查环境 =============
echo 检查环境...

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] Python 未安装
    pause
    exit /b 1
)
echo [成功] Python 已安装

REM 检查 GPU
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>nul
if errorlevel 1 (
    echo [警告] nvidia-smi 不可用
) else (
    echo.
    echo GPU 信息:
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | findstr /n "^" | findstr "^1:"
)

REM 检查 PyTorch
python -c "import torch; print('✅ PyTorch:', torch.__version__)" 2>nul
if errorlevel 1 (
    echo [错误] PyTorch 未安装
    pause
    exit /b 1
)

REM 检查 CUDA
python -c "import torch; print('✅ CUDA 可用:', torch.cuda.is_available())"

REM 检查数据文件
if not exist "%DATA_PATH%" (
    echo [警告] 数据文件不存在: %DATA_PATH%
    echo        请确认数据路径是否正确
)

REM ============= 开始训练 =============
echo.
echo ==========================================
echo 开始训练...
echo 按 Ctrl+C 可以停止训练
echo ==========================================
echo.

REM 开始训练并保存日志
python trainer\train_pretrain.py ^
    --device cuda:0 ^
    --dtype %DTYPE% ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --accumulation_steps %ACCUMULATION_STEPS% ^
    --learning_rate %LEARNING_RATE% ^
    --grad_clip 1.0 ^
    --save_interval %SAVE_INTERVAL% ^
    --log_interval %LOG_INTERVAL% ^
    --save_dir %SAVE_DIR% ^
    --save_weight %MODEL_NAME% ^
    --hidden_size %HIDDEN_SIZE% ^
    --num_hidden_layers %NUM_LAYERS% ^
    --max_seq_len %MAX_SEQ_LEN% ^
    --data_path %DATA_PATH% ^
    --from_resume 1 ^
    --num_workers 4 ^
    2>&1 | tee %LOG_FILE%

REM ============= 训练完成 =============
echo.
echo ==========================================
echo      训练完成！
echo ==========================================
echo 完成时间: %date% %time%
echo 模型保存位置: %SAVE_DIR%
echo 日志文件: %LOG_FILE%
echo.

REM 显示保存的模型
echo 已保存的模型:
dir /b %SAVE_DIR%\*.pth 2>nul
dir /b %SAVE_DIR%\*.safetensors 2>nul

REM 转换为 safetensors 格式
echo.
if exist "%SAVE_DIR%\%MODEL_NAME%_%HIDDEN_SIZE%.pth" (
    echo 转换为 SafeTensors 格式...
    if exist "scripts\convert_to_safetensors.py" (
        python scripts\convert_to_safetensors.py "%SAVE_DIR%\%MODEL_NAME%_%HIDDEN_SIZE%.pth"
    ) else (
        echo [警告] 转换脚本不存在
    )
)

REM 清理pip缓存（可选）
echo.
set /p cleanup="是否清理pip缓存以释放空间? (y/N): "
if /i "%cleanup%"=="y" (
    echo 清理pip缓存...
    pip cache purge
    echo 缓存已清理
)

echo.
echo ==========================================
echo 全部完成！
echo ==========================================
pause

