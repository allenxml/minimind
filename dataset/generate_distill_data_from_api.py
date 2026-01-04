"""
从 OpenRouter API 生成蒸馏训练数据

本脚本支持从主流大模型API生成蒸馏数据，用于知识蒸馏训练。

核心特性:
1. 支持 OpenRouter 平台的所有模型
2. 自动检测模型是否支持推理过程输出
3. 支持批量生成和断点续传
4. 自动格式化为标准训练数据格式

支持的输出模式:
- answer_only: 仅获取答案（用于标准知识蒸馏）
- reasoning: 获取推理过程+答案（用于推理蒸馏）

使用方法:
    # 标准蒸馏数据生成（使用顶级模型）
    python dataset/generate_distill_data_from_api.py \
        --api_key YOUR_OPENROUTER_KEY \
        --model anthropic/claude-opus-4.5 \
        --input_file dataset/sft_mini_512.jsonl \
        --output_file dataset/distill_opus45_512.jsonl \
        --mode answer_only
    
    # 推理蒸馏数据生成（高性价比）
    python dataset/generate_distill_data_from_api.py \
        --api_key YOUR_OPENROUTER_KEY \
        --model deepseek/deepseek-r1 \
        --input_file dataset/sft_mini_512.jsonl \
        --output_file dataset/distill_r1_512.jsonl \
        --mode reasoning
"""

import os
import sys
import json
import time
import argparse
import requests
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 支持推理过程的模型列表 (基于 OpenRouter 2025 排名)
REASONING_MODELS = [
    # 顶级推理模型
    'anthropic/claude-opus-4.5',
    'anthropic/claude-sonnet-4.5',
    'openai/gpt-5.2',
    # 推理专用模型
    'deepseek/deepseek-r1',
    'deepseek/deepseek-reasoner',
    'openai/o1-preview',
    'openai/o1-mini',
    'openai/o3-mini',
    # 其他
    'anthropic/claude-3-opus',
    'anthropic/claude-3.5-sonnet',
]


class OpenRouterClient:
    """OpenRouter API 客户端"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        """
        初始化 OpenRouter 客户端
        
        Args:
            api_key: OpenRouter API 密钥
            base_url: API 基础 URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jingyaogong/minimind",
            "X-Title": "MiniMind Distillation"
        }
    
    def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7,
        max_tokens: int = 2048,
        reasoning_effort: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        调用 OpenRouter 的 chat completion API
        
        Args:
            model: 模型名称（如 'openai/gpt-4'）
            messages: 对话消息列表
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            reasoning_effort: 推理强度（对支持的模型如 o1，可选 'low'/'medium'/'high'）
            
        Returns:
            (回答内容, 推理过程或None)
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # 如果模型支持推理强度参数
        if reasoning_effort and any(m in model.lower() for m in ['o1', 'o3', 'deepseek-r1', 'reasoner']):
            payload["reasoning_effort"] = reasoning_effort
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            choice = result['choices'][0]
            message = choice['message']
            
            # 提取回答内容
            content = message.get('content', '')
            
            # 提取推理过程（某些模型会在单独的字段返回）
            reasoning = None
            if 'reasoning_content' in message:
                reasoning = message['reasoning_content']
            elif 'thoughts' in message:
                reasoning = message['thoughts']
            
            return content, reasoning
            
        except requests.exceptions.RequestException as e:
            print(f"API 调用失败: {e}")
            return "", None
        except KeyError as e:
            print(f"响应格式错误: {e}")
            print(f"完整响应: {response.text if 'response' in locals() else 'N/A'}")
            return "", None
    
    def is_reasoning_model(self, model: str) -> bool:
        """
        检查模型是否支持推理过程输出
        
        Args:
            model: 模型名称
            
        Returns:
            是否支持推理过程
        """
        return any(rm in model.lower() for rm in REASONING_MODELS)


def load_jsonl(file_path: str) -> List[Dict]:
    """
    加载 JSONL 文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """
    保存为 JSONL 文件
    
    Args:
        data: 数据列表
        file_path: 文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def append_jsonl(item: Dict, file_path: str):
    """
    追加一条数据到 JSONL 文件
    
    Args:
        item: 数据项
        file_path: 文件路径
    """
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_conversation_from_item(item: Dict) -> List[Dict[str, str]]:
    """
    从数据项中提取对话消息
    
    支持多种数据格式:
    1. {"conversations": [{"role": "...", "content": "..."}, ...]}
    2. {"instruction": "...", "input": "...", "output": "..."}
    3. {"prompt": "...", "response": "..."}
    
    Args:
        item: 数据项
        
    Returns:
        OpenAI 格式的消息列表
    """
    # 格式 1: conversations 字段
    if 'conversations' in item:
        messages = []
        for msg in item['conversations']:
            role = msg.get('from', msg.get('role', 'user'))
            # 统一 role 格式
            if role in ['human', 'user']:
                role = 'user'
            elif role in ['gpt', 'assistant', 'bot']:
                role = 'assistant'
            elif role == 'system':
                role = 'system'
            
            content = msg.get('value', msg.get('content', ''))
            messages.append({"role": role, "content": content})
        
        # 移除最后的 assistant 消息（这是我们要让教师模型生成的）
        if messages and messages[-1]['role'] == 'assistant':
            messages = messages[:-1]
        
        return messages
    
    # 格式 2: instruction/input/output
    elif 'instruction' in item:
        messages = []
        instruction = item['instruction']
        input_text = item.get('input', '')
        
        if input_text:
            content = f"{instruction}\n\n{input_text}"
        else:
            content = instruction
        
        messages.append({"role": "user", "content": content})
        return messages
    
    # 格式 3: prompt/response
    elif 'prompt' in item:
        return [{"role": "user", "content": item['prompt']}]
    
    else:
        raise ValueError(f"不支持的数据格式: {item.keys()}")


def format_reasoning_output(content: str, reasoning: Optional[str]) -> str:
    """
    格式化推理过程输出为统一格式
    
    Args:
        content: 回答内容
        reasoning: 推理过程（可选）
        
    Returns:
        格式化后的输出
    """
    if reasoning:
        return f"<think>{reasoning}</think>\n<answer>{content}</answer>"
    else:
        return content


def generate_distill_data(
    client: OpenRouterClient,
    input_file: str,
    output_file: str,
    model: str,
    mode: str = 'answer_only',
    temperature: float = 0.7,
    max_tokens: int = 2048,
    reasoning_effort: Optional[str] = None,
    max_samples: Optional[int] = None,
    resume: bool = True,
    rate_limit_delay: float = 0.5
):
    """
    生成蒸馏训练数据
    
    Args:
        client: OpenRouter 客户端
        input_file: 输入数据文件
        output_file: 输出数据文件
        model: 教师模型名称
        mode: 输出模式（'answer_only' 或 'reasoning'）
        temperature: 温度参数
        max_tokens: 最大生成 token 数
        reasoning_effort: 推理强度
        max_samples: 最大样本数（None 表示全部）
        resume: 是否断点续传
        rate_limit_delay: API 调用间隔（秒）
    """
    # 加载输入数据
    print(f"正在加载输入数据: {input_file}")
    input_data = load_jsonl(input_file)
    
    if max_samples:
        input_data = input_data[:max_samples]
    
    print(f"共 {len(input_data)} 条数据")
    
    # 检查模型是否支持推理过程
    is_reasoning_model = client.is_reasoning_model(model)
    if mode == 'reasoning' and not is_reasoning_model:
        print(f"警告: 模型 {model} 可能不支持推理过程输出")
        print(f"支持推理过程的模型: {REASONING_MODELS}")
        user_input = input("是否继续？(y/n): ")
        if user_input.lower() != 'y':
            return
    
    # 断点续传: 检查已生成的数据
    processed_count = 0
    if resume and os.path.exists(output_file):
        existing_data = load_jsonl(output_file)
        processed_count = len(existing_data)
        print(f"检测到已生成 {processed_count} 条数据，从第 {processed_count + 1} 条开始")
    
    # 生成数据
    print(f"\n开始生成蒸馏数据...")
    print(f"教师模型: {model}")
    print(f"输出模式: {mode}")
    print(f"推理强度: {reasoning_effort or 'N/A'}")
    
    success_count = 0
    error_count = 0
    
    for idx, item in enumerate(tqdm(input_data, desc="生成中")):
        # 跳过已处理的数据
        if idx < processed_count:
            continue
        
        try:
            # 提取输入消息
            messages = extract_conversation_from_item(item)
            
            # 调用教师模型
            content, reasoning = client.chat_completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort
            )
            
            if not content:
                print(f"\n第 {idx + 1} 条数据生成失败（空响应）")
                error_count += 1
                continue
            
            # 格式化输出
            if mode == 'reasoning':
                output = format_reasoning_output(content, reasoning)
            else:
                output = content
            
            # 构建输出数据项
            output_item = {
                "conversations": [
                    *messages,
                    {"role": "assistant", "content": output}
                ]
            }
            
            # 添加元数据
            output_item['metadata'] = {
                'teacher_model': model,
                'mode': mode,
                'has_reasoning': reasoning is not None,
                'source_index': idx
            }
            
            # 追加到输出文件
            append_jsonl(output_item, output_file)
            success_count += 1
            
            # API 速率限制
            time.sleep(rate_limit_delay)
            
        except Exception as e:
            print(f"\n第 {idx + 1} 条数据处理出错: {e}")
            error_count += 1
            continue
    
    print(f"\n数据生成完成!")
    print(f"成功: {success_count} 条")
    print(f"失败: {error_count} 条")
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 OpenRouter API 生成蒸馏训练数据")
    
    # API 配置
    parser.add_argument("--api_key", type=str, required=True,
                        help="OpenRouter API 密钥")
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1",
                        help="API 基础 URL")
    parser.add_argument("--model", type=str, required=True,
                        help="教师模型名称（如 'openai/gpt-4', 'deepseek/deepseek-r1'）")
    
    # 数据配置
    parser.add_argument("--input_file", type=str, required=True,
                        help="输入数据文件路径（JSONL 格式）")
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出数据文件路径（JSONL 格式）")
    parser.add_argument("--mode", type=str, choices=['answer_only', 'reasoning'], default='answer_only',
                        help="输出模式: answer_only（仅答案）或 reasoning（推理过程+答案）")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="温度参数（推荐 0.7-1.0）")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="最大生成 token 数")
    parser.add_argument("--reasoning_effort", type=str, choices=['low', 'medium', 'high'],
                        help="推理强度（仅支持部分模型如 o1/o3/deepseek-r1）")
    parser.add_argument("--max_samples", type=int,
                        help="最大样本数（用于测试）")
    parser.add_argument("--no_resume", action="store_true",
                        help="禁用断点续传")
    parser.add_argument("--rate_limit_delay", type=float, default=0.5,
                        help="API 调用间隔（秒）")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = OpenRouterClient(api_key=args.api_key, base_url=args.base_url)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 生成数据
    generate_distill_data(
        client=client,
        input_file=args.input_file,
        output_file=args.output_file,
        model=args.model,
        mode=args.mode,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        max_samples=args.max_samples,
        resume=not args.no_resume,
        rate_limit_delay=args.rate_limit_delay
    )

