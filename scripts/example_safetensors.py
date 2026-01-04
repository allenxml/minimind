#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SafeTensors 使用示例

本脚本演示如何在 MiniMind 中使用 SafeTensors 格式
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import (
    init_model, 
    save_model_safetensors, 
    load_model_safetensors,
    SAFETENSORS_AVAILABLE
)


def example_1_check_availability():
    """示例1: 检查 SafeTensors 是否可用"""
    print("\n" + "="*60)
    print("示例1: 检查 SafeTensors 可用性")
    print("="*60)
    
    if SAFETENSORS_AVAILABLE:
        print("✅ SafeTensors 已安装并可用")
        from safetensors import __version__
        print(f"   版本: {__version__}")
    else:
        print("❌ SafeTensors 未安装")
        print("   安装方法: pip install safetensors")


def example_2_save_model():
    """示例2: 保存模型为 SafeTensors 格式"""
    print("\n" + "="*60)
    print("示例2: 保存模型为 SafeTensors 格式")
    print("="*60)
    
    # 创建一个小模型
    config = MiniMindConfig(hidden_size=256, num_hidden_layers=4)
    model = MiniMindForCausalLM(config)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 保存为 SafeTensors
    output_path = "out/example_model.safetensors"
    os.makedirs("out", exist_ok=True)
    
    success = save_model_safetensors(model, output_path, half_precision=True)
    
    if success:
        file_size = os.path.getsize(output_path) / (1024**2)
        print(f"✅ 模型已保存: {output_path}")
        print(f"   文件大小: {file_size:.2f} MB")
    
    return output_path


def example_3_load_model():
    """示例3: 从 SafeTensors 加载模型"""
    print("\n" + "="*60)
    print("示例3: 从 SafeTensors 加载模型")
    print("="*60)
    
    if not SAFETENSORS_AVAILABLE:
        print("⚠️  SafeTensors 未安装，跳过此示例")
        return
    
    # 确保有模型文件
    model_path = "out/example_model.safetensors"
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        print("   运行示例2先创建模型")
        return
    
    # 创建模型架构
    config = MiniMindConfig(hidden_size=256, num_hidden_layers=4)
    model = MiniMindForCausalLM(config)
    
    # 加载权重
    state_dict = load_model_safetensors(model_path, device='cpu')
    model.load_state_dict(state_dict)
    
    print(f"✅ 模型加载成功")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


def example_4_compare_formats():
    """示例4: 对比 PyTorch 和 SafeTensors 格式"""
    print("\n" + "="*60)
    print("示例4: 对比不同格式的性能")
    print("="*60)
    
    import time
    
    # 创建模型
    config = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
    model = MiniMindForCausalLM(config)
    
    os.makedirs("out", exist_ok=True)
    
    # 保存为 PyTorch 格式
    pth_path = "out/compare_model.pth"
    print("\n📝 保存为 PyTorch 格式...")
    start = time.time()
    torch.save(model.state_dict(), pth_path)
    pth_save_time = time.time() - start
    pth_size = os.path.getsize(pth_path) / (1024**2)
    print(f"   保存时间: {pth_save_time:.3f}秒")
    print(f"   文件大小: {pth_size:.2f} MB")
    
    # 保存为 SafeTensors 格式
    if SAFETENSORS_AVAILABLE:
        safe_path = "out/compare_model.safetensors"
        print("\n📝 保存为 SafeTensors 格式...")
        start = time.time()
        save_model_safetensors(model, safe_path, half_precision=False)
        safe_save_time = time.time() - start
        safe_size = os.path.getsize(safe_path) / (1024**2)
        print(f"   保存时间: {safe_save_time:.3f}秒")
        print(f"   文件大小: {safe_size:.2f} MB")
        
        # 加载性能对比
        print("\n📂 加载性能对比...")
        
        # 加载 PyTorch
        start = time.time()
        _ = torch.load(pth_path, map_location='cpu')
        pth_load_time = time.time() - start
        print(f"   PyTorch 加载时间: {pth_load_time:.3f}秒")
        
        # 加载 SafeTensors
        start = time.time()
        _ = load_model_safetensors(safe_path, device='cpu')
        safe_load_time = time.time() - start
        print(f"   SafeTensors 加载时间: {safe_load_time:.3f}秒")
        
        # 总结
        print("\n" + "="*60)
        print("📊 性能对比总结")
        print("="*60)
        print(f"保存时间: SafeTensors 比 PyTorch 快 {pth_save_time/safe_save_time:.2f}x")
        print(f"加载时间: SafeTensors 比 PyTorch 快 {pth_load_time/safe_load_time:.2f}x ⚡")
        print(f"文件大小: 基本相同 ({abs(safe_size-pth_size):.2f} MB 差异)")


def example_5_auto_load():
    """示例5: 使用 init_model 自动选择格式"""
    print("\n" + "="*60)
    print("示例5: 自动选择最佳格式加载")
    print("="*60)
    
    # 确保有测试文件
    if not os.path.exists("out/compare_model.safetensors"):
        print("⚠️  需要先运行示例4创建测试文件")
        return
    
    config = MiniMindConfig(hidden_size=512, num_hidden_layers=8)
    
    # 优先加载 SafeTensors（默认）
    print("\n🔍 测试1: 优先加载 SafeTensors")
    try:
        model, _ = init_model(
            config,
            from_weight='compare_model',
            save_dir='out',
            tokenizer_path='model',
            device='cpu',
            prefer_safetensors=True  # 默认
        )
        print("✅ 成功加载（优先 SafeTensors）")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
    
    # 仅加载 PyTorch 格式
    print("\n🔍 测试2: 仅加载 PyTorch 格式")
    try:
        model, _ = init_model(
            config,
            from_weight='compare_model',
            save_dir='out',
            tokenizer_path='model',
            device='cpu',
            prefer_safetensors=False
        )
        print("✅ 成功加载（PyTorch 格式）")
    except Exception as e:
        print(f"❌ 加载失败: {e}")


def example_6_convert_existing():
    """示例6: 转换现有模型"""
    print("\n" + "="*60)
    print("示例6: 使用转换脚本")
    print("="*60)
    
    print("""
转换现有模型的方法：

1. 转换单个文件:
   python scripts/convert_to_safetensors.py out/pretrain_512.pth

2. 批量转换:
   python scripts/convert_to_safetensors.py --batch out/

3. 指定输出路径:
   python scripts/convert_to_safetensors.py out/model.pth --output models/model.safetensors

4. 跳过验证（更快）:
   python scripts/convert_to_safetensors.py out/model.pth --no-verify

详细用法请参考: SafeTensors使用指南.md
    """)


def main():
    """运行所有示例"""
    print("""
╔════════════════════════════════════════════════════════════╗
║         SafeTensors 使用示例                               ║
║         MiniMind Project                                   ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    # 运行所有示例
    example_1_check_availability()
    
    if SAFETENSORS_AVAILABLE:
        example_2_save_model()
        example_3_load_model()
        example_4_compare_formats()
        example_5_auto_load()
    else:
        print("\n⚠️  SafeTensors 未安装，部分示例跳过")
        print("   安装方法: pip install safetensors")
    
    example_6_convert_existing()
    
    print("\n" + "="*60)
    print("✅ 所有示例完成")
    print("="*60)
    print("\n详细文档:")
    print("  - SafeTensors使用指南.md")
    print("  - SafeTensors快速参考.txt")


if __name__ == "__main__":
    main()

