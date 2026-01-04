"""
MiniMind 交互式对话推理脚本

本脚本提供了一个命令行交互界面，用于与 MiniMind 模型进行对话。

主要功能:
1. 加载训练好的模型权重
2. 提供交互式对话界面
3. 支持流式输出（逐字显示）
4. 支持多轮对话（保持上下文）
5. 支持 LoRA 权重加载

使用方法:
    # 基础对话
    python infer_chat.py
    
    # 使用特定模型
    python infer_chat.py --model_path ../out/full_sft_512.pth
    
    # 使用 LoRA
    python infer_chat.py --lora_path ../out/lora/lora_identity_512.pth
    
    # 推理模式（带思考过程）
    python infer_chat.py --reasoning 1

交互命令:
- 输入问题后按回车发送
- 输入 'clear' 清空对话历史
- 输入 'exit' 或 'quit' 退出程序
- Ctrl+C 强制退出
"""

import os
import sys

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora


def load_model(args):
    """
    加载模型和 tokenizer
    
    Args:
        args: 命令行参数
        
    Returns:
        Tuple[model, tokenizer]: 加载好的模型和分词器
    """
    # 创建模型配置
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        flash_attn=args.flash_attn
    )
    
    # 加载 tokenizer
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 创建模型
    model = MiniMindForCausalLM(lm_config)
    
    # 加载模型权重
    if os.path.exists(args.model_path):
        print(f"📂 加载模型权重: {args.model_path}")
        weights = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(weights, strict=False)
    else:
        print(f"⚠️  未找到模型权重: {args.model_path}")
        print("   使用随机初始化的模型")
    
    # 加载 LoRA 权重（如果指定）
    if args.lora_path and os.path.exists(args.lora_path):
        print(f"📂 加载 LoRA 权重: {args.lora_path}")
        apply_lora(model)
        load_lora(model, args.lora_path)
    
    # 移动到设备并设置为评估模式
    model = model.to(args.device)
    model.eval()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型参数量: {total_params / 1e6:.2f} M")
    
    return model, tokenizer


def generate_response(model, tokenizer, messages, args):
    """
    生成模型回复
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        messages: 对话历史
        args: 命令行参数
        
    Returns:
        str: 模型生成的回复
    """
    # 使用 chat_template 格式化对话
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).to(args.device)
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码生成的 token
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def stream_generate(model, tokenizer, messages, args):
    """
    流式生成模型回复（逐字输出）
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        messages: 对话历史
        args: 命令行参数
        
    Yields:
        str: 每次生成的新 token
    """
    # 使用 chat_template 格式化对话
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).to(args.device)
    
    input_ids = inputs['input_ids']
    past_key_values = None
    generated_tokens = []
    
    # 逐 token 生成
    for _ in range(args.max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        # 获取下一个 token 的 logits
        next_token_logits = outputs.logits[:, -1, :]
        
        # 采样或贪婪解码
        if args.do_sample:
            # 应用温度
            next_token_logits = next_token_logits / args.temperature
            
            # Top-k 采样
            if args.top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, args.top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p 采样
            if args.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > args.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # 贪婪解码
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # 检查是否生成了 EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
        
        # 解码并输出
        token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
        generated_tokens.append(next_token.item())
        yield token_text
        
        # 更新输入
        input_ids = next_token
        past_key_values = outputs.past_key_values
    
    return ''.join(tokenizer.decode(generated_tokens, skip_special_tokens=True))


def chat_loop(model, tokenizer, args):
    """
    交互式对话循环
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        args: 命令行参数
    """
    print("\n" + "="*60)
    print("🤖 MiniMind 对话系统")
    print("="*60)
    print("💡 提示:")
    print("   - 输入问题后按回车发送")
    print("   - 输入 'clear' 清空对话历史")
    print("   - 输入 'exit' 或 'quit' 退出")
    print("="*60 + "\n")
    
    # 初始化对话历史
    messages = []
    
    # 添加系统提示（如果有）
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    
    while True:
        try:
            # 获取用户输入
            user_input = input("👤 用户: ").strip()
            
            # 处理特殊命令
            if not user_input:
                continue
            elif user_input.lower() in ['exit', 'quit']:
                print("👋 再见！")
                break
            elif user_input.lower() == 'clear':
                messages = []
                if args.system_prompt:
                    messages.append({"role": "system", "content": args.system_prompt})
                print("🗑️  对话历史已清空\n")
                continue
            
            # 添加用户消息
            messages.append({"role": "user", "content": user_input})
            
            # 生成回复
            print("🤖 助手: ", end="", flush=True)
            
            if args.stream:
                # 流式输出
                response_parts = []
                for token in stream_generate(model, tokenizer, messages, args):
                    print(token, end="", flush=True)
                    response_parts.append(token)
                response = ''.join(response_parts)
            else:
                # 一次性输出
                response = generate_response(model, tokenizer, messages, args)
                print(response)
            
            print("\n")
            
            # 添加助手回复到历史
            messages.append({"role": "assistant", "content": response})
            
            # 限制历史长度（防止上下文过长）
            if len(messages) > args.max_history * 2 + 1:  # +1 for system prompt
                # 保留系统提示和最近的对话
                if args.system_prompt:
                    messages = messages[:1] + messages[-(args.max_history * 2):]
                else:
                    messages = messages[-(args.max_history * 2):]
                    
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind 交互式对话")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="../out/full_sft_512.pth",
                        help="模型权重路径")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA 权重路径（可选）")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8,
                        help="隐藏层数量")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1],
                        help="是否使用 MoE 架构")
    parser.add_argument("--flash_attn", type=int, default=1, choices=[0, 1],
                        help="是否使用 Flash Attention")
    
    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="最大生成 token 数")
    parser.add_argument("--do_sample", type=int, default=1, choices=[0, 1],
                        help="是否使用采样（0=贪婪解码）")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p 采样参数")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k 采样参数")
    
    # 对话参数
    parser.add_argument("--system_prompt", type=str, default="你是一个有帮助的AI助手。",
                        help="系统提示词")
    parser.add_argument("--max_history", type=int, default=10,
                        help="保留的最大对话轮数")
    parser.add_argument("--stream", type=int, default=1, choices=[0, 1],
                        help="是否流式输出")
    parser.add_argument("--reasoning", type=int, default=0, choices=[0, 1],
                        help="是否使用推理模式")
    
    # 设备参数
    parser.add_argument("--device", type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="运行设备")
    
    args = parser.parse_args()
    
    # 推理模式的系统提示
    if args.reasoning:
        args.system_prompt = "你是一个善于思考的AI助手。在回答问题时，请先在<think>标签中进行思考，然后在<answer>标签中给出最终答案。"
    
    # 加载模型
    model, tokenizer = load_model(args)
    
    # 开始对话
    chat_loop(model, tokenizer, args)