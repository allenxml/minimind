#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU兼容性检测脚本
用于检查当前环境是否支持sm_120等新架构
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.gpu_utils import print_gpu_info, get_gpu_compute_capability, check_flash_attention_support, get_recommended_dtype


def check_pytorch_version():
    """检查PyTorch版本"""
    print("\n" + "="*60)
    print("🔍 PyTorch 环境检测")
    print("="*60)
    
    print(f"📌 PyTorch版本: {torch.__version__}")
    print(f"📌 CUDA可用: {'✅ 是' if torch.cuda.is_available() else '❌ 否'}")
    
    if torch.cuda.is_available():
        print(f"📌 CUDA版本: {torch.version.cuda}")
        print(f"📌 cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"📌 GPU数量: {torch.cuda.device_count()}")
    else:
        print("\n⚠️  警告: 未检测到CUDA，将使用CPU模式")
        print("   如果您有NVIDIA GPU，请检查CUDA安装")
        return False
    
    return True


def check_dependencies():
    """检查其他依赖"""
    print("\n" + "="*60)
    print("🔍 依赖包检测")
    print("="*60)
    
    deps = {
        'transformers': 'transformers',
        'numpy': 'numpy',
        'torch': 'torch'
    }
    
    for name, package in deps.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name}: {version}")
        except ImportError:
            print(f"❌ {name}: 未安装")


def run_simple_test():
    """运行简单的CUDA测试"""
    if not torch.cuda.is_available():
        return
    
    print("\n" + "="*60)
    print("🧪 CUDA 功能测试")
    print("="*60)
    
    try:
        # 测试基本张量操作
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print("✅ 基本CUDA张量运算: 正常")
        
        # 测试数据类型
        if torch.cuda.is_bf16_supported():
            x_bf16 = x.to(torch.bfloat16)
            y_bf16 = y.to(torch.bfloat16)
            z_bf16 = torch.matmul(x_bf16, y_bf16)
            print("✅ BFloat16运算: 正常")
        else:
            print("⚠️  BFloat16: 不支持")
        
        # 测试Flash Attention
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            q = torch.randn(2, 8, 32, 64).cuda()
            k = torch.randn(2, 8, 32, 64).cuda()
            v = torch.randn(2, 8, 32, 64).cuda()
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            print("✅ Flash Attention (SDPA): 可用")
        else:
            print("⚠️  Flash Attention: 不可用")
            
    except Exception as e:
        print(f"❌ CUDA测试失败: {e}")


def check_compute_capability_support():
    """检查计算能力支持情况"""
    if not torch.cuda.is_available():
        return
    
    major, minor = get_gpu_compute_capability()
    
    print("\n" + "="*60)
    print("🎯 架构支持情况")
    print("="*60)
    
    # 架构映射
    arch_info = {
        (12, 0): ("Blackwell", "✅ 最新架构", "RTX 5090, 5080"),
        (9, 0): ("Hopper", "✅ 数据中心级", "H100, H200"),
        (8, 9): ("Ada Lovelace", "✅ 消费级旗舰", "RTX 4090, 4080, 4070"),
        (8, 6): ("Ampere", "✅ 消费级", "RTX 3090, 3080, 3070"),
        (8, 0): ("Ampere", "✅ 数据中心级", "A100, A10"),
        (7, 5): ("Turing", "✅ 较老但支持", "RTX 2080, T4"),
        (7, 0): ("Volta", "✅ 较老但支持", "V100"),
    }
    
    arch_name, support, examples = arch_info.get(
        (major, minor), 
        ("未知架构", "⚠️  可能需要更新PyTorch", "")
    )
    
    print(f"📌 架构: {arch_name}")
    print(f"📌 支持状态: {support}")
    if examples:
        print(f"📌 代表GPU: {examples}")
    
    # 检查是否需要升级
    if major >= 12:
        torch_version = torch.__version__.split('+')[0]
        major_v, minor_v = map(int, torch_version.split('.')[:2])
        if major_v < 2 or (major_v == 2 and minor_v < 7):
            print("\n⚠️  警告: Blackwell架构需要PyTorch >= 2.7.0")
            print("   当前版本可能无法完全利用GPU性能")
            print("   建议运行: pip install --upgrade torch>=2.7.0 torchvision>=0.22.0")
        else:
            print("\n✅ PyTorch版本适配Blackwell架构！")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🚀 MiniMind GPU 兼容性检测工具")
    print("="*60)
    print("本工具用于检测GPU架构和环境配置")
    print("特别支持最新的Blackwell架构 (sm_120)")
    
    # 1. 检查PyTorch
    if not check_pytorch_version():
        print("\n❌ 无法继续检测，请先安装CUDA和PyTorch")
        print("   安装指南: https://pytorch.org/get-started/locally/")
        return
    
    # 2. 检查GPU详细信息
    print_gpu_info()
    
    # 3. 检查架构支持
    check_compute_capability_support()
    
    # 4. 检查依赖
    check_dependencies()
    
    # 5. 运行功能测试
    run_simple_test()
    
    # 总结
    print("\n" + "="*60)
    print("✅ 检测完成")
    print("="*60)
    
    if torch.cuda.is_available():
        major, minor = get_gpu_compute_capability()
        flash_support = check_flash_attention_support()
        
        print("\n📋 配置建议:")
        print(f"   - dtype: {get_recommended_dtype()}")
        print(f"   - flash_attn: {flash_support}")
        
        if major >= 12:
            print("\n💡 针对Blackwell架构的建议:")
            print("   1. 使用bfloat16可获得最佳性能")
            print("   2. 适当增加batch_size以充分利用显存带宽")
            print("   3. 直接运行训练脚本，程序会自动适配")
        
        print("\n🚀 可以开始训练了！运行:")
        print("   python trainer/train_pretrain.py --epochs 1 --batch_size 32")
    else:
        print("\n⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()

