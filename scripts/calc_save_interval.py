#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算训练保存间隔工具

根据期望的保存时间间隔，计算应该设置的 save_interval 步数
"""

import argparse


def calculate_save_interval(time_minutes, batch_size=32, accumulation_steps=8, 
                           dataset_size=100000, time_per_step_seconds=1.5):
    """
    计算达到指定保存时间间隔所需的步数
    
    Args:
        time_minutes: 期望的保存时间间隔（分钟）
        batch_size: 批次大小
        accumulation_steps: 梯度累积步数
        dataset_size: 数据集样本数
        time_per_step_seconds: 每步平均耗时（秒）
    
    Returns:
        int: 建议的 save_interval 值
    """
    target_seconds = time_minutes * 60
    save_interval = int(target_seconds / time_per_step_seconds)
    
    # 计算每个epoch的步数
    steps_per_epoch = dataset_size // (batch_size * accumulation_steps)
    saves_per_epoch = steps_per_epoch / save_interval if save_interval > 0 else 0
    
    print("="*70)
    print(f"{'保存间隔计算工具':^70}")
    print("="*70)
    print(f"\n📋 输入参数:")
    print(f"   目标保存间隔:     {time_minutes} 分钟")
    print(f"   Batch Size:       {batch_size}")
    print(f"   梯度累积步数:     {accumulation_steps}")
    print(f"   有效 Batch Size:  {batch_size * accumulation_steps}")
    print(f"   数据集大小:       {dataset_size:,} 样本")
    print(f"   预计每步耗时:     {time_per_step_seconds} 秒")
    
    print(f"\n📊 计算结果:")
    print(f"   建议 save_interval:  {save_interval} 步")
    print(f"   实际保存间隔:        约 {save_interval * time_per_step_seconds / 60:.1f} 分钟")
    print(f"   每 epoch 步数:       {steps_per_epoch} 步")
    print(f"   每 epoch 保存次数:   约 {saves_per_epoch:.1f} 次")
    
    print(f"\n💡 使用建议:")
    print(f"   python3 trainer/train_pretrain.py \\")
    print(f"       --batch_size {batch_size} \\")
    print(f"       --accumulation_steps {accumulation_steps} \\")
    print(f"       --save_interval {save_interval} \\")
    print(f"       --device cuda:0 \\")
    print(f"       --dtype bfloat16")
    
    print("\n"+"="*70)
    
    return save_interval


def estimate_training_time(epochs, dataset_size, batch_size, accumulation_steps, 
                          time_per_step_seconds=1.5):
    """
    估算总训练时间
    
    Args:
        epochs: 训练轮数
        dataset_size: 数据集大小
        batch_size: 批次大小
        accumulation_steps: 梯度累积
        time_per_step_seconds: 每步耗时
    """
    steps_per_epoch = dataset_size // (batch_size * accumulation_steps)
    total_steps = steps_per_epoch * epochs
    total_seconds = total_steps * time_per_step_seconds
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    
    print(f"\n⏱️  训练时间估算:")
    print(f"   总步数:       {total_steps:,} 步")
    print(f"   预计总时间:   约 {hours} 小时 {minutes} 分钟")
    print(f"   每 epoch:     约 {steps_per_epoch * time_per_step_seconds / 60:.1f} 分钟")


def main():
    parser = argparse.ArgumentParser(
        description="计算训练保存间隔",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 每30分钟保存一次
  python3 scripts/calc_save_interval.py --time 30
  
  # 每1小时保存，batch_size=64
  python3 scripts/calc_save_interval.py --time 60 --batch-size 64
  
  # 完整参数
  python3 scripts/calc_save_interval.py \\
      --time 30 \\
      --batch-size 32 \\
      --accumulation 8 \\
      --dataset-size 100000 \\
      --step-time 1.5
        """
    )
    
    parser.add_argument("--time", type=float, default=30,
                       help="期望的保存时间间隔（分钟），默认: 30")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="批次大小，默认: 32")
    parser.add_argument("--accumulation", type=int, default=8,
                       help="梯度累积步数，默认: 8")
    parser.add_argument("--dataset-size", type=int, default=100000,
                       help="数据集样本数，默认: 100000")
    parser.add_argument("--step-time", type=float, default=1.5,
                       help="每步平均耗时（秒），默认: 1.5")
    parser.add_argument("--epochs", type=int, default=None,
                       help="训练轮数（用于估算总时间）")
    
    args = parser.parse_args()
    
    # 计算保存间隔
    save_interval = calculate_save_interval(
        time_minutes=args.time,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation,
        dataset_size=args.dataset_size,
        time_per_step_seconds=args.step_time
    )
    
    # 估算训练时间
    if args.epochs:
        estimate_training_time(
            epochs=args.epochs,
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation,
            time_per_step_seconds=args.step_time
        )
    
    print()


if __name__ == "__main__":
    main()

