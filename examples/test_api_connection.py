"""
测试 OpenRouter API 连接

使用此脚本测试你的 OpenRouter API Key 是否有效，
以及查看可用的模型列表。

使用方法:
    python test_api_connection.py YOUR_OPENROUTER_KEY
"""

import sys
import requests
import json


def test_api_connection(api_key: str):
    """测试 API 连接并列出可用模型"""
    
    print("=" * 70)
    print("OpenRouter API 连接测试")
    print("=" * 70)
    
    # 测试 1: 获取模型列表
    print("\n[测试 1] 获取可用模型列表...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        
        models = response.json()
        print(f"✅ 成功! 共找到 {len(models.get('data', []))} 个可用模型")
        
        # 显示推荐的模型（基于 OpenRouter 2025 排名）
        print("\n🏆 推荐用于蒸馏的顶级模型:")
        recommended = [
            "anthropic/claude-opus-4.5",
            "anthropic/claude-sonnet-4.5",
            "google/gemini-3-flash-preview",
            "openai/gpt-5.2",
            "deepseek/deepseek-v3.2",
            "deepseek/deepseek-r1",
        ]
        
        for model_id in recommended:
            # 查找模型信息
            model_info = None
            for m in models.get('data', []):
                if m['id'] == model_id:
                    model_info = m
                    break
            
            if model_info:
                pricing = model_info.get('pricing', {})
                prompt_price = float(pricing.get('prompt', 0)) * 1000000  # 转换为 $/1M tokens
                completion_price = float(pricing.get('completion', 0)) * 1000000
                
                print(f"\n  📦 {model_id}")
                print(f"     输入: ${prompt_price:.2f}/1M tokens")
                print(f"     输出: ${completion_price:.2f}/1M tokens")
            else:
                print(f"\n  ⚠️  {model_id} (未找到)")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 失败: {e}")
        return False
    
    # 测试 2: 简单的 API 调用
    print("\n" + "=" * 70)
    print("[测试 2] 测试简单的 API 调用...")
    
    payload = {
        "model": "google/gemini-2.5-flash",  # 使用快速低成本模型测试
        "messages": [
            {"role": "user", "content": "你好，请用一句话介绍你自己。"}
        ],
        "max_tokens": 100
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                **headers,
                "HTTP-Referer": "https://github.com/jingyaogong/minimind",
                "X-Title": "MiniMind Test"
            },
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        print(f"✅ 成功!")
        print(f"\n模型响应: {content}")
        
        # 显示使用统计
        usage = result.get('usage', {})
        if usage:
            print(f"\nToken 使用:")
            print(f"  输入: {usage.get('prompt_tokens', 0)} tokens")
            print(f"  输出: {usage.get('completion_tokens', 0)} tokens")
            print(f"  总计: {usage.get('total_tokens', 0)} tokens")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"响应内容: {e.response.text}")
        return False
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过! API Key 有效且可以正常使用。")
    print("=" * 70)
    print("\n下一步:")
    print("1. 查看文档: docs/API蒸馏训练指南.md")
    print("2. 运行示例: bash examples/api_distillation_example.sh")
    print("3. 生成数据: python scripts/generate_distill_data_from_api.py --help")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python test_api_connection.py YOUR_OPENROUTER_KEY")
        print("\n获取 API Key: https://openrouter.ai/")
        sys.exit(1)
    
    api_key = sys.argv[1]
    
    if not api_key or api_key == "YOUR_OPENROUTER_KEY":
        print("错误: 请提供有效的 OpenRouter API Key")
        print("\n获取 API Key: https://openrouter.ai/")
        sys.exit(1)
    
    success = test_api_connection(api_key)
    sys.exit(0 if success else 1)

