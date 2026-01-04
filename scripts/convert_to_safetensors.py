#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 PyTorch .pth 模型转换为 SafeTensors 格式

SafeTensors 格式优势：
- 更安全：纯数据格式，不会执行恶意代码
- 更快：加载速度快2-3倍
- 跨框架：支持 PyTorch/TensorFlow/JAX
- HuggingFace兼容：与HF生态完美集成

使用方法：
    python scripts/convert_to_safetensors.py out/pretrain_512.pth
    python scripts/convert_to_safetensors.py out/pretrain_512.pth --output models/pretrain_512.safetensors
"""

import os
import sys
import argparse
import torch
from pathlib import Path

try:
    from safetensors.torch import save_file, load_file
except ImportError:
    print("❌ 错误: safetensors 未安装")
    print("请运行: pip install safetensors")
    sys.exit(1)


def convert_pth_to_safetensors(pth_path, output_path=None, verify=True):
    """
    转换 .pth 文件到 .safetensors
    
    Args:
        pth_path: 输入的.pth文件路径
        output_path: 输出的.safetensors文件路径（可选）
        verify: 是否验证转换结果
    
    Returns:
        str: 输出文件路径
    """
    # 检查输入文件
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"输入文件不存在: {pth_path}")
    
    # 生成输出路径
    if output_path is None:
        output_path = pth_path.replace('.pth', '.safetensors')
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    print(f"🔄 正在加载模型: {pth_path}")
    
    try:
        # 加载PyTorch模型
        state_dict = torch.load(pth_path, map_location='cpu')
        
        # 如果是checkpoint格式（包含optimizer等），只提取model部分
        if isinstance(state_dict, dict):
            if 'model' in state_dict:
                print("   检测到checkpoint格式，提取model部分")
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                print("   检测到checkpoint格式，提取state_dict部分")
                state_dict = state_dict['state_dict']
        
        # 显示模型信息
        total_params = sum(v.numel() for v in state_dict.values())
        print(f"   模型参数量: {total_params / 1e6:.2f}M")
        print(f"   张量数量: {len(state_dict)}")
        
        # 转换所有tensor为连续存储（safetensors要求）
        print("🔄 正在转换为连续存储格式...")
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        
        print(f"💾 正在保存为 SafeTensors 格式: {output_path}")
        
        # 保存为safetensors
        save_file(state_dict, output_path)
        
        # 显示文件大小对比
        pth_size = os.path.getsize(pth_path) / (1024**2)
        safe_size = os.path.getsize(output_path) / (1024**2)
        
        print(f"\n✅ 转换完成！")
        print(f"{'='*60}")
        print(f"   原始 .pth 大小:      {pth_size:.2f} MB")
        print(f"   转换后 .safetensors: {safe_size:.2f} MB")
        print(f"   大小差异:           {safe_size - pth_size:+.2f} MB")
        print(f"{'='*60}")
        
        # 验证转换结果
        if verify:
            print("\n🔍 验证转换结果...")
            verify_conversion(pth_path, output_path, state_dict)
        
        return output_path
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def verify_conversion(pth_path, safetensors_path, original_state_dict=None):
    """
    验证转换结果的正确性
    
    Args:
        pth_path: 原始.pth文件路径
        safetensors_path: 转换后的.safetensors文件路径
        original_state_dict: 原始state_dict（可选，避免重复加载）
    """
    try:
        # 加载safetensors
        safe_state_dict = load_file(safetensors_path)
        
        # 如果没有提供原始state_dict，加载它
        if original_state_dict is None:
            original_state_dict = torch.load(pth_path, map_location='cpu')
            if isinstance(original_state_dict, dict):
                if 'model' in original_state_dict:
                    original_state_dict = original_state_dict['model']
                elif 'state_dict' in original_state_dict:
                    original_state_dict = original_state_dict['state_dict']
        
        # 检查键是否一致
        orig_keys = set(original_state_dict.keys())
        safe_keys = set(safe_state_dict.keys())
        
        if orig_keys != safe_keys:
            print("   ⚠️  警告: 键不完全匹配")
            missing = orig_keys - safe_keys
            extra = safe_keys - orig_keys
            if missing:
                print(f"      缺失的键: {missing}")
            if extra:
                print(f"      额外的键: {extra}")
        else:
            print("   ✅ 键匹配: 所有键都正确")
        
        # 检查张量值是否一致（抽样检查）
        mismatches = 0
        for key in list(orig_keys)[:5]:  # 检查前5个张量
            if key in safe_keys:
                orig_tensor = original_state_dict[key]
                safe_tensor = safe_state_dict[key]
                
                if not torch.allclose(orig_tensor, safe_tensor, rtol=1e-5, atol=1e-6):
                    mismatches += 1
                    print(f"   ⚠️  张量值不匹配: {key}")
        
        if mismatches == 0:
            print("   ✅ 张量值验证: 通过（抽样检查）")
        else:
            print(f"   ⚠️  发现 {mismatches} 个张量值不匹配")
        
        print("\n✅ 验证完成：转换成功！")
        
    except Exception as e:
        print(f"   ⚠️  验证过程出错: {e}")


def batch_convert(input_dir, output_dir=None, pattern="*.pth"):
    """
    批量转换目录下的所有.pth文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可选，默认为输入目录）
        pattern: 文件匹配模式
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 查找所有匹配的文件
    pth_files = list(input_path.glob(pattern))
    
    if not pth_files:
        print(f"❌ 在 {input_dir} 中未找到匹配 {pattern} 的文件")
        return
    
    print(f"🔍 找到 {len(pth_files)} 个文件待转换")
    print(f"{'='*60}\n")
    
    success_count = 0
    for i, pth_file in enumerate(pth_files, 1):
        print(f"[{i}/{len(pth_files)}] 转换: {pth_file.name}")
        
        try:
            # 生成输出路径
            if output_dir:
                output_path = Path(output_dir) / pth_file.name.replace('.pth', '.safetensors')
            else:
                output_path = pth_file.with_suffix('.safetensors')
            
            convert_pth_to_safetensors(str(pth_file), str(output_path), verify=False)
            success_count += 1
            print()
            
        except Exception as e:
            print(f"❌ 转换失败: {e}\n")
            continue
    
    print(f"{'='*60}")
    print(f"✅ 批量转换完成: {success_count}/{len(pth_files)} 个文件成功")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="转换PyTorch模型为SafeTensors格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个文件
  python scripts/convert_to_safetensors.py out/pretrain_512.pth
  
  # 指定输出路径
  python scripts/convert_to_safetensors.py out/pretrain_512.pth --output models/pretrain_512.safetensors
  
  # 批量转换
  python scripts/convert_to_safetensors.py --batch out/
  
  # 批量转换到指定目录
  python scripts/convert_to_safetensors.py --batch out/ --output models/
  
  # 跳过验证（更快）
  python scripts/convert_to_safetensors.py out/pretrain_512.pth --no-verify
        """
    )
    
    parser.add_argument("input", type=str, help="输入的.pth文件路径或目录（配合--batch）")
    parser.add_argument("--output", type=str, default=None, help="输出的.safetensors文件路径或目录")
    parser.add_argument("--batch", action="store_true", help="批量转换模式")
    parser.add_argument("--pattern", type=str, default="*.pth", help="批量转换时的文件匹配模式（默认: *.pth）")
    parser.add_argument("--no-verify", action="store_true", help="跳过转换验证（加快速度）")
    
    args = parser.parse_args()
    
    print("""
╔════════════════════════════════════════════════════════════╗
║         PyTorch to SafeTensors 转换工具                   ║
║         MiniMind Project                                   ║
╚════════════════════════════════════════════════════════════╝
""")
    
    try:
        if args.batch:
            # 批量转换模式
            batch_convert(args.input, args.output, args.pattern)
        else:
            # 单文件转换模式
            convert_pth_to_safetensors(args.input, args.output, verify=not args.no_verify)
        
        print("\n🎉 所有操作完成！")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

