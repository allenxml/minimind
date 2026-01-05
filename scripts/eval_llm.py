"""
MiniMind 模型评估与推理脚本

本脚本提供了 MiniMind 模型的评估和交互式对话功能。
支持多种模型权重、LoRA 适配器和生成参数配置。

主要功能:
1. 加载不同阶段的模型权重（pretrain, full_sft, dpo, reason 等）
2. 支持 LoRA 权重加载（身份认同、医疗领域等）
3. 支持自动测试和手动输入两种模式
4. 支持流式输出（逐字显示）
5. 支持多轮对话（保持上下文历史）
6. 支持 RoPE 位置编码外推（处理长序列）

模型权重说明:
- pretrain: 预训练权重，只学习了语言模型
- full_sft: 全参数监督微调权重，学习了对话能力
- dpo: DPO 对齐后的权重，更符合人类偏好
- reason: 推理蒸馏权重，具有思考能力
- grpo/spo: 强化学习优化后的权重

使用方法:
    # 基础对话（使用 full_sft 权重）
    python scripts/eval_llm.py
    
    # 使用推理模型
    python scripts/eval_llm.py --weight reason
    
    # 使用 LoRA 权重
    python scripts/eval_llm.py --lora_weight lora_identity
    
    # 使用 MoE 架构
    python scripts/eval_llm.py --hidden_size 640 --use_moe 1
    
    # 启用 RoPE 外推（处理长序列）
    python scripts/eval_llm.py --inference_rope_scaling

交互模式:
- 模式 0: 自动测试，使用预设的测试问题
- 模式 1: 手动输入，用户自由提问
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import warnings
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed

# 忽略警告信息
warnings.filterwarnings('ignore')


def init_model(args):
    """
    初始化模型和分词器
    
    根据 args.load_from 参数决定加载方式:
    - 'model': 加载原生 PyTorch 权重（MiniMind 格式）
    - 其他路径: 使用 HuggingFace transformers 加载
    
    Args:
        args: 命令行参数，包含:
            - load_from: 模型加载路径
            - hidden_size: 隐藏层维度
            - num_hidden_layers: 隐藏层数量
            - use_moe: 是否使用 MoE 架构
            - inference_rope_scaling: 是否启用 RoPE 外推
            - save_dir: 权重保存目录
            - weight: 权重名称前缀
            - lora_weight: LoRA 权重名称
            - device: 运行设备
            
    Returns:
        Tuple[model, tokenizer]: 加载好的模型和分词器
        
    模型加载流程:
    1. 加载 tokenizer
    2. 根据配置创建模型
    3. 加载模型权重
    4. 可选：加载 LoRA 权重
    5. 设置为评估模式并移动到目标设备
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    
    if 'model' in args.load_from:
        # ========== 加载 MiniMind 原生格式 ==========
        # 创建模型配置
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling  # RoPE 外推配置
        ))
        
        # 构建权重文件路径
        moe_suffix = '_moe' if args.use_moe else ''
        # 处理绝对路径和相对路径
        save_dir = args.save_dir if os.path.isabs(args.save_dir) else f'./{args.save_dir}'
        ckp = f'{save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        
        # 加载模型权重
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        
        # 加载 LoRA 权重（如果指定）
        if args.lora_weight != 'None':
            # 先应用 LoRA 结构
            apply_lora(model)
            # 再加载 LoRA 权重
            lora_path = f'{save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth'
            load_lora(model, lora_path)
    else:
        # ========== 加载 HuggingFace 格式 ==========
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    
    # 打印模型参数量
    print(f'MiniMind模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    
    # 设置为评估模式并移动到目标设备
    return model.eval().to(args.device), tokenizer


def main():
    """
    主函数：解析参数并运行对话循环
    
    工作流程:
    1. 解析命令行参数
    2. 初始化模型和分词器
    3. 选择运行模式（自动测试/手动输入）
    4. 进入对话循环
    """
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    
    # 模型加载参数
    parser.add_argument('--load_from', default='model', type=str, 
                        help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, 
                        help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, 
                        help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, 
                        help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    
    # 模型架构参数
    parser.add_argument('--hidden_size', default=512, type=int, 
                        help="隐藏层维度（512=Small-26M, 640=MoE-145M, 768=Base-104M）")
    parser.add_argument('--num_hidden_layers', default=8, type=int, 
                        help="隐藏层数量（Small/MoE=8, Base=16）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], 
                        help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', 
                        help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    
    # 生成参数
    parser.add_argument('--max_new_tokens', default=8192, type=int, 
                        help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, 
                        help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, 
                        help="nucleus采样阈值（0-1）")
    
    # 对话参数
    parser.add_argument('--historys', default=0, type=int, 
                        help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    
    # 设备参数
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, 
                        help="运行设备")
    
    args = parser.parse_args()
    
    # ========== 预设测试问题 ==========
    # 这些问题覆盖了不同类型的任务，用于快速评估模型能力
    prompts = [
        '你有什么特长？',                              # 自我介绍
        '为什么天空是蓝色的',                          # 科学解释
        '请用Python写一个计算斐波那契数列的函数',      # 代码生成
        '解释一下"光合作用"的基本过程',                # 知识问答
        '如果明天下雨，我应该如何出门',                # 推理判断
        '比较一下猫和狗作为宠物的优缺点',              # 对比分析
        '解释什么是机器学习',                          # 概念解释
        '推荐一些中国的美食'                           # 推荐任务
    ]
    
    # ========== 初始化 ==========
    conversation = []  # 对话历史
    model, tokenizer = init_model(args)
    
    # 选择运行模式
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    
    # 创建流式输出器（逐字显示生成结果）
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # ========== 对话循环 ==========
    # 根据模式选择问题来源
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('👶: '), '')
    
    for prompt in prompt_iter:
        # 设置随机种子（确保可重复性）
        setup_seed(2026)  # 或使用 setup_seed(random.randint(0, 2048)) 获得随机结果
        
        # 自动测试模式下打印问题
        if input_mode == 0: 
            print(f'👶: {prompt}')
        
        # 管理对话历史
        # 只保留最近 args.historys 轮对话（每轮包含 user 和 assistant 两条消息）
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        # ========== 构建输入 ==========
        # 使用 chat_template 格式化对话
        templates = {
            "conversation": conversation, 
            "tokenize": False, 
            "add_generation_prompt": True
        }
        
        # 推理模型特殊处理：启用思考模式
        if args.weight == 'reason': 
            templates["enable_thinking"] = True
        
        # 根据权重类型选择输入格式
        if args.weight != 'pretrain':
            # SFT/DPO/Reason 等模型使用 chat_template
            inputs = tokenizer.apply_chat_template(**templates)
        else:
            # 预训练模型使用简单格式
            inputs = tokenizer.bos_token + prompt
        
        # Tokenize 输入
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        # ========== 生成回复 ==========
        print('🤖️: ', end='')
        
        # 调用模型生成
        generated_ids = model.generate(
            inputs=inputs["input_ids"],           # 输入 token ID
            attention_mask=inputs["attention_mask"],  # 注意力掩码
            max_new_tokens=args.max_new_tokens,   # 最大生成长度
            do_sample=True,                        # 使用采样
            streamer=streamer,                     # 流式输出
            pad_token_id=tokenizer.pad_token_id,  # 填充 token ID
            eos_token_id=tokenizer.eos_token_id,  # 结束 token ID
            top_p=args.top_p,                      # Top-p 采样
            temperature=args.temperature,          # 温度参数
            repetition_penalty=1.0                 # 重复惩罚（1.0 表示不惩罚）
        )
        
        # 解码生成的回复（去除输入部分）
        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=True
        )
        
        # 将回复添加到对话历史
        conversation.append({"role": "assistant", "content": response})
        
        print('\n\n')


if __name__ == "__main__":
    main()

