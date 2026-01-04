"""
GPU 兼容性工具

本模块提供 GPU 检测和兼容性配置功能，特别支持最新的 GPU 架构。

主要功能:
1. 检测 GPU 计算能力（Compute Capability）
2. 自动配置 Flash Attention 支持
3. 推荐最佳数据类型（dtype）
4. 确保模型配置与 GPU 兼容

支持的 GPU 架构:
- Blackwell (sm_120): RTX 5090, 5080 等
- Hopper (sm_90): H100, H200 等
- Ada Lovelace (sm_89): RTX 4090, 4080, 4070 等
- Ampere (sm_80/86): RTX 3090, 3080, A100 等
- Turing (sm_75): RTX 2080, T4 等
- Volta (sm_70): V100 等

使用方法:
    from model.gpu_utils import ensure_gpu_compatibility
    
    config = MiniMindConfig(...)
    config = ensure_gpu_compatibility(config)  # 自动配置
"""

import torch


def get_gpu_compute_capability():
    """
    获取当前 GPU 的计算能力（Compute Capability）
    
    计算能力是 NVIDIA GPU 的版本标识，格式为 (major, minor)。
    不同的计算能力对应不同的 GPU 架构和功能支持。
    
    计算能力对照表:
    - 12.0: Blackwell (RTX 5090, 5080)
    - 9.0: Hopper (H100, H200)
    - 8.9: Ada Lovelace (RTX 4090, 4080, 4070)
    - 8.6: Ampere 消费级 (RTX 3090, 3080, 3070)
    - 8.0: Ampere 数据中心 (A100, A10)
    - 7.5: Turing (RTX 2080, T4)
    - 7.0: Volta (V100)
    
    Returns:
        Tuple[int, int]: (major, minor) 计算能力版本
                        如果没有 GPU 或检测失败，返回 (0, 0)
    
    Example:
        >>> major, minor = get_gpu_compute_capability()
        >>> print(f"GPU 计算能力: {major}.{minor}")
        GPU 计算能力: 8.9
    """
    if not torch.cuda.is_available():
        return (0, 0)
    
    try:
        # 获取当前设备的计算能力
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        return (major, minor)
    except Exception as e:
        print(f"警告: 无法获取 GPU 计算能力: {e}")
        return (0, 0)


def check_flash_attention_support():
    """
    检查当前环境是否支持 Flash Attention
    
    Flash Attention 是一种高效的注意力计算实现，可以:
    1. 减少内存使用（从 O(n²) 降低到 O(n)）
    2. 加速计算（通过减少内存访问）
    3. 支持更长的序列
    
    支持条件:
    1. PyTorch >= 2.0（包含 scaled_dot_product_attention）
    2. CUDA 可用
    3. GPU 计算能力 >= 7.0（Volta 及以上）
    
    Returns:
        bool: 是否支持 Flash Attention
        
    Example:
        >>> if check_flash_attention_support():
        ...     print("可以使用 Flash Attention")
    """
    # 检查 PyTorch 版本是否支持 SDPA
    if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        return False
    
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        return False
    
    # 检查 GPU 计算能力
    major, minor = get_gpu_compute_capability()
    
    # Flash Attention 需要 Volta (7.0) 及以上架构
    if major < 7:
        return False
    
    return True


def get_recommended_dtype():
    """
    根据 GPU 能力推荐最佳数据类型
    
    不同的 GPU 架构对不同数据类型有不同的支持和性能:
    
    - bfloat16: 
      - 优点: 动态范围大，训练稳定
      - 支持: Ampere (8.0) 及以上
      - 推荐用于: 训练和推理
      
    - float16:
      - 优点: 广泛支持，内存效率高
      - 支持: 所有现代 GPU
      - 注意: 可能需要损失缩放
      
    - float32:
      - 优点: 最高精度
      - 支持: 所有 GPU
      - 缺点: 内存使用多，速度慢
    
    Returns:
        str: 推荐的数据类型 ('bfloat16', 'float16', 或 'float32')
        
    Example:
        >>> dtype = get_recommended_dtype()
        >>> print(f"推荐使用: {dtype}")
    """
    if not torch.cuda.is_available():
        return 'float32'
    
    # 检查 bfloat16 支持
    if torch.cuda.is_bf16_supported():
        return 'bfloat16'
    
    # 检查 GPU 计算能力
    major, minor = get_gpu_compute_capability()
    
    # Ampere (8.0) 及以上支持 bfloat16
    if major >= 8:
        return 'bfloat16'
    
    # 其他 GPU 使用 float16
    if major >= 7:
        return 'float16'
    
    # 较老的 GPU 使用 float32
    return 'float32'


def print_gpu_info():
    """
    打印详细的 GPU 信息
    
    输出包括:
    - GPU 名称和数量
    - 计算能力
    - 显存大小
    - CUDA 版本
    - Flash Attention 支持状态
    - 推荐的数据类型
    
    Example:
        >>> print_gpu_info()
        ============================================================
        🔍 GPU 信息
        ============================================================
        📌 GPU 数量: 1
        📌 GPU 0: NVIDIA GeForce RTX 4090
           - 计算能力: 8.9
           - 显存: 24.0 GB
        📌 CUDA 版本: 12.1
        📌 Flash Attention: ✅ 支持
        📌 推荐 dtype: bfloat16
        ============================================================
    """
    print("\n" + "="*60)
    print("🔍 GPU 信息")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        print("="*60)
        return
    
    # GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"📌 GPU 数量: {gpu_count}")
    
    # 每个 GPU 的详细信息
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        major, minor = props.major, props.minor
        
        print(f"📌 GPU {i}: {props.name}")
        print(f"   - 计算能力: {major}.{minor}")
        print(f"   - 显存: {memory_gb:.1f} GB")
    
    # CUDA 版本
    print(f"📌 CUDA 版本: {torch.version.cuda}")
    
    # Flash Attention 支持
    flash_support = check_flash_attention_support()
    flash_status = "✅ 支持" if flash_support else "❌ 不支持"
    print(f"📌 Flash Attention: {flash_status}")
    
    # 推荐的数据类型
    recommended_dtype = get_recommended_dtype()
    print(f"📌 推荐 dtype: {recommended_dtype}")
    
    print("="*60)


def ensure_gpu_compatibility(config):
    """
    确保模型配置与当前 GPU 兼容
    
    根据 GPU 能力自动调整配置:
    1. 检测 Flash Attention 支持并配置
    2. 对于不支持的 GPU，禁用某些高级特性
    3. 打印兼容性信息
    
    Args:
        config: MiniMindConfig 配置对象
        
    Returns:
        config: 调整后的配置对象
        
    Example:
        >>> config = MiniMindConfig(flash_attn=True)
        >>> config = ensure_gpu_compatibility(config)
        >>> # 如果 GPU 不支持 Flash Attention，会自动禁用
    """
    if not torch.cuda.is_available():
        # CPU 模式: 禁用 Flash Attention
        if hasattr(config, 'flash_attn'):
            config.flash_attn = False
        print("⚠️  CUDA 不可用，使用 CPU 模式")
        return config
    
    # 获取 GPU 信息
    major, minor = get_gpu_compute_capability()
    
    # 检查 Flash Attention 支持
    flash_support = check_flash_attention_support()
    
    if hasattr(config, 'flash_attn'):
        if config.flash_attn and not flash_support:
            print(f"⚠️  GPU (sm_{major}{minor}) 不支持 Flash Attention，已自动禁用")
            config.flash_attn = False
        elif config.flash_attn and flash_support:
            print(f"✅ Flash Attention 已启用 (GPU: sm_{major}{minor})")
    
    # 特殊架构处理
    if major >= 12:
        # Blackwell 架构 (sm_120)
        print(f"🚀 检测到 Blackwell 架构 (sm_{major}{minor})")
        print("   建议使用 bfloat16 以获得最佳性能")
    elif major >= 9:
        # Hopper 架构 (sm_90)
        print(f"🚀 检测到 Hopper 架构 (sm_{major}{minor})")
    elif major >= 8:
        # Ampere/Ada 架构
        if minor >= 9:
            print(f"✅ 检测到 Ada Lovelace 架构 (sm_{major}{minor})")
        else:
            print(f"✅ 检测到 Ampere 架构 (sm_{major}{minor})")
    
    return config


def get_optimal_batch_size(hidden_size, seq_len, gpu_memory_gb=None):
    """
    根据 GPU 显存估算最优批次大小
    
    这是一个粗略估算，实际最优值可能需要通过实验确定。
    
    估算公式（简化）:
    memory_per_sample ≈ 4 * hidden_size * seq_len * num_layers * 2 (前向+反向)
    
    Args:
        hidden_size (int): 模型隐藏层维度
        seq_len (int): 序列长度
        gpu_memory_gb (float): GPU 显存大小（GB），None 则自动检测
        
    Returns:
        int: 推荐的批次大小
        
    Example:
        >>> batch_size = get_optimal_batch_size(512, 512)
        >>> print(f"推荐批次大小: {batch_size}")
    """
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = props.total_memory / (1024**3)
        else:
            gpu_memory_gb = 8  # 默认假设 8GB
    
    # 预留一部分显存给系统和其他开销
    available_memory_gb = gpu_memory_gb * 0.7
    
    # 粗略估算每个样本的内存使用（字节）
    # 这是一个非常简化的估算
    bytes_per_sample = hidden_size * seq_len * 4 * 8  # 假设 8 层，float32
    
    # 计算批次大小
    batch_size = int(available_memory_gb * 1024**3 / bytes_per_sample)
    
    # 限制在合理范围内
    batch_size = max(1, min(batch_size, 128))
    
    # 对齐到 2 的幂次（可选，有时能提高效率）
    # batch_size = 2 ** int(math.log2(batch_size))
    
    return batch_size
