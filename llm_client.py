# llm_client.py
"""
统一调用大模型客户端（支持 Qwen + Qianfan OpenAI 兼容模式）
作者：AI 开发者
日期：2026年2月
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI, APIError, Timeout, RateLimitError

load_dotenv()

# ==============================
# 1. 通义千问 (Qwen) - 原生 API
# ==============================
def call_qwen(prompt: str, model="qwen-max") -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return "❌ 缺少 DASHSCOPE_API_KEY，请检查 .env 文件"

    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=api_key,
        timeout=30
    )
    
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except (APIError, Timeout, RateLimitError) as e:
            err_msg = str(e)
            print(f"  → Qwen 尝试 {i+1} 失败: {err_msg[:100]}")
            if i < 2:
                time.sleep(2)
        except Exception as e:
            return f"❌ Qwen 调用异常: {e}"
    
    return "❌ Qwen 所有重试失败"


# ==============================
# 2. 百度千帆 (Qianfan) - OpenAI 兼容模式
# ==============================
def call_qianfan(prompt: str, model="ernie-3.5-8k") -> str:
    api_key = os.getenv("QIANFAN_OPENAI_API_KEY")
    if not api_key:
        return "❌ 缺少 QIANFAN_OPENAI_API_KEY，请检查 .env 文件"

    # 模型名称映射（确保使用官方支持的名称）
    valid_models = {
        "ernie-3.5-8k",
        "ernie-speed-8k",
        "ernie-4.0-8k",
        "ernie-4.5-turbo-128k",
    }
    if model not in valid_models:
        return f"❌ 不支持的百度模型: '{model}'。请选择: {sorted(valid_models)}"

    client = OpenAI(
        base_url="https://qianfan.baidubce.com/v2",
        api_key=api_key,
        timeout=30
    )
    
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.9
            )
            return response.choices[0].message.content
            
        except (APIError, Timeout, RateLimitError) as e:
            err_msg = str(e)
            print(f"  → Qianfan 尝试 {i+1} 失败: {err_msg[:100]}")
            if i < 2:
                time.sleep(2)
        except Exception as e:
            return f"❌ Qianfan 调用异常: {e}"
    
    return "❌ Qianfan 所有重试失败"


# ==============================
# 3. 统一入口函数
# ==============================
def call_llm(provider: str, prompt: str, model: str = None) -> str:
    """
    统一调用大模型
    Args:
        provider: 'qwen' 或 'qianfan'
        prompt: 用户输入
        model: 模型名称（可选）
    """
    print(f"🚀 调用 {provider.upper()} ...")
    
    providers = {
        "qwen": (call_qwen, "qwen-max"),
        "qianfan": (call_qianfan, "ernie-3.5-8k"),
    }

    if provider not in providers:
        return f"❌ 不支持的模型提供商: {provider}（支持: qwen, qianfan）"

    func, default_model = providers[provider]
    actual_model = model or default_model
    
    if actual_model != default_model:
        print(f"   → 使用模型: {actual_model}")
    
    return func(prompt, actual_model)