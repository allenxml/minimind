"""
从 API 生成问答对（大模型自己生成问题和答案）

本脚本让大模型自己生成高质量的问答数据，用于训练。

与 generate_distill_data_from_api.py 的区别:
- generate_distill_data_from_api.py: 用户提供问题 → API生成答案
- generate_qa_pairs_from_api.py: API自己生成问题 + 答案

使用场景:
1. 没有现成的问题数据集
2. 想要扩充特定领域的数据
3. 需要大模型生成多样化的训练数据

使用方法:
    # 生成通用问答数据
    python dataset/generate_qa_pairs_from_api.py \
        --api_key YOUR_KEY \
        --model anthropic/claude-opus-4.5 \
        --topic "Python编程" \
        --num_samples 1000 \
        --output_file dataset/qa_python.jsonl
    
    # 生成带推理过程的数学题
    python dataset/generate_qa_pairs_from_api.py \
        --api_key YOUR_KEY \
        --model deepseek/deepseek-r1 \
        --topic "初中数学应用题" \
        --mode reasoning \
        --num_samples 500 \
        --output_file dataset/qa_math_reasoning.jsonl
"""

import os
import sys
import json
import time
import argparse
import requests
from tqdm import tqdm
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OpenRouterClient:
    """OpenRouter API 客户端"""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/jingyaogong/minimind",
            "X-Title": "MiniMind QA Generation"
        }
    
    def generate_qa_pair(
        self,
        model: str,
        topic: str,
        difficulty: str = "medium",
        language: str = "zh",
        mode: str = "answer_only",
        temperature: float = 0.8
    ) -> Dict:
        """
        生成一个问答对
        
        Args:
            model: 模型名称
            topic: 主题（如"Python编程"、"高中物理"）
            difficulty: 难度（easy/medium/hard）
            language: 语言（zh/en）
            mode: 模式（answer_only/reasoning）
            temperature: 温度参数
            
        Returns:
            {"question": "...", "answer": "..."}
        """
        # 构建提示词
        if language == "zh":
            difficulty_map = {"easy": "简单", "medium": "中等", "hard": "困难"}
            difficulty_text = difficulty_map.get(difficulty, "中等")
            
            if mode == "reasoning":
                prompt = f"""请生成一个关于"{topic}"的{difficulty_text}难度问题和详细答案。

要求:
1. 问题要有实际意义，能够考察理解和思维能力
2. 答案要包含详细的推理过程
3. 使用以下格式回答:

问题：[在这里写问题]

答案：
<think>
[在这里写详细的推理步骤]
</think>
<answer>
[在这里写最终答案]
</answer>

现在请生成一个新的问题和答案："""
            else:
                prompt = f"""请生成一个关于"{topic}"的{difficulty_text}难度问题和答案。

要求:
1. 问题要清晰明确
2. 答案要准确完整
3. 使用以下格式回答:

问题：[在这里写问题]

答案：[在这里写答案]

现在请生成一个新的问题和答案："""
        else:
            difficulty_text = difficulty
            if mode == "reasoning":
                prompt = f"""Generate a {difficulty_text} difficulty question and detailed answer about "{topic}".

Requirements:
1. Question should be meaningful and test understanding
2. Answer should include detailed reasoning process
3. Use this format:

Question: [write question here]

Answer:
<think>
[write detailed reasoning steps here]
</think>
<answer>
[write final answer here]
</answer>

Please generate:"""
            else:
                prompt = f"""Generate a {difficulty_text} difficulty question and answer about "{topic}".

Requirements:
1. Question should be clear
2. Answer should be accurate
3. Use this format:

Question: [write question here]

Answer: [write answer here]

Please generate:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # 解析问题和答案
            qa_pair = self._parse_qa_pair(content, language)
            return qa_pair
            
        except Exception as e:
            print(f"API 调用失败: {e}")
            return None
    
    def _parse_qa_pair(self, content: str, language: str) -> Dict:
        """解析生成的问答对"""
        qa_pair = {"question": "", "answer": ""}
        
        # 尝试解析格式
        if language == "zh":
            # 中文格式
            if "问题：" in content and "答案：" in content:
                parts = content.split("答案：")
                question_part = parts[0].replace("问题：", "").strip()
                answer_part = parts[1].strip() if len(parts) > 1 else ""
                
                qa_pair["question"] = question_part
                qa_pair["answer"] = answer_part
            else:
                # 容错处理
                lines = content.strip().split("\n")
                if len(lines) >= 2:
                    qa_pair["question"] = lines[0].strip()
                    qa_pair["answer"] = "\n".join(lines[1:]).strip()
        else:
            # 英文格式
            if "Question:" in content and "Answer:" in content:
                parts = content.split("Answer:")
                question_part = parts[0].replace("Question:", "").strip()
                answer_part = parts[1].strip() if len(parts) > 1 else ""
                
                qa_pair["question"] = question_part
                qa_pair["answer"] = answer_part
            else:
                lines = content.strip().split("\n")
                if len(lines) >= 2:
                    qa_pair["question"] = lines[0].strip()
                    qa_pair["answer"] = "\n".join(lines[1:]).strip()
        
        return qa_pair


def generate_qa_dataset(
    client: OpenRouterClient,
    model: str,
    topic: str,
    num_samples: int,
    output_file: str,
    difficulty: str = "medium",
    language: str = "zh",
    mode: str = "answer_only",
    temperature: float = 0.8,
    rate_limit_delay: float = 1.0,
    resume: bool = True
):
    """
    生成问答数据集
    
    Args:
        client: OpenRouter 客户端
        model: 模型名称
        topic: 主题
        num_samples: 生成数量
        output_file: 输出文件
        difficulty: 难度
        language: 语言
        mode: 模式
        temperature: 温度
        rate_limit_delay: API 调用间隔
        resume: 是否断点续传
    """
    # 检查已生成的数量
    start_idx = 0
    if resume and os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_lines = f.readlines()
            start_idx = len(existing_lines)
        print(f"检测到已生成 {start_idx} 条数据，从第 {start_idx + 1} 条开始")
    
    print(f"\n开始生成问答数据...")
    print(f"模型: {model}")
    print(f"主题: {topic}")
    print(f"难度: {difficulty}")
    print(f"语言: {language}")
    print(f"模式: {mode}")
    print(f"目标数量: {num_samples}")
    
    success_count = 0
    error_count = 0
    
    for idx in tqdm(range(start_idx, num_samples), desc="生成中"):
        try:
            # 生成问答对
            qa_pair = client.generate_qa_pair(
                model=model,
                topic=topic,
                difficulty=difficulty,
                language=language,
                mode=mode,
                temperature=temperature
            )
            
            if not qa_pair or not qa_pair.get("question") or not qa_pair.get("answer"):
                print(f"\n第 {idx + 1} 条数据生成失败（空响应）")
                error_count += 1
                continue
            
            # 构建标准格式
            item = {
                "conversations": [
                    {"role": "user", "content": qa_pair["question"]},
                    {"role": "assistant", "content": qa_pair["answer"]}
                ],
                "metadata": {
                    "topic": topic,
                    "difficulty": difficulty,
                    "language": language,
                    "mode": mode,
                    "teacher_model": model,
                    "index": idx
                }
            }
            
            # 追加到文件
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
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
    parser = argparse.ArgumentParser(description="从 API 生成问答数据集")
    
    # API 配置
    parser.add_argument("--api_key", type=str, required=True,
                        help="OpenRouter API 密钥")
    parser.add_argument("--model", type=str, required=True,
                        help="教师模型名称")
    
    # 数据配置
    parser.add_argument("--topic", type=str, required=True,
                        help="数据主题（如'Python编程'、'初中数学'）")
    parser.add_argument("--num_samples", type=int, required=True,
                        help="生成数量")
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出文件路径")
    
    # 生成参数
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard"],
                        help="难度级别")
    parser.add_argument("--language", type=str, default="zh",
                        choices=["zh", "en"],
                        help="语言")
    parser.add_argument("--mode", type=str, default="answer_only",
                        choices=["answer_only", "reasoning"],
                        help="模式")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="温度参数（推荐0.8-1.0以增加多样性）")
    parser.add_argument("--rate_limit_delay", type=float, default=1.0,
                        help="API 调用间隔")
    parser.add_argument("--no_resume", action="store_true",
                        help="禁用断点续传")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = OpenRouterClient(api_key=args.api_key)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 生成数据
    generate_qa_dataset(
        client=client,
        model=args.model,
        topic=args.topic,
        num_samples=args.num_samples,
        output_file=args.output_file,
        difficulty=args.difficulty,
        language=args.language,
        mode=args.mode,
        temperature=args.temperature,
        rate_limit_delay=args.rate_limit_delay,
        resume=not args.no_resume
    )

