# qwen_client.py
"""
健壮的 Qwen 调用客户端（Day 3 实践）
支持：认证、限流处理、重试、错误分类
"""

import os
import time
import logging
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError, RateLimitError, AuthenticationError

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("QwenClient")

# 初始化客户端（全局复用）
client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    timeout=30
)

def call_qwen(
    prompt: str,
    model: str = "qwen-max",
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> str:
    """
    调用通义千问大模型
    
    Args:
        prompt: 用户输入
        model: 模型名称（默认 qwen-max）
        max_retries: 最大重试次数
        retry_delay: 初始重试延迟（秒），使用指数退避
        
    Returns:
        模型返回的文本，或错误信息
    """
    if not client.api_key:
        return "❌ 错误：未配置 DASHSCOPE_API_KEY，请检查 .env 文件"

    for attempt in range(max_retries + 1):
        try:
            logger.info(f"🚀 调用 Qwen (尝试 {attempt + 1}/{max_retries + 1})")
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            result = response.choices[0].message.content.strip()
            logger.info("✅ Qwen 调用成功")
            return result

        except AuthenticationError as e:
            return f"❌ 认证失败 (401): API Key 无效或缺失。请检查 .env 中的 DASHSCOPE_API_KEY"

        except RateLimitError as e:
            # 处理限流：等待后重试
            wait_time = retry_delay * (2 ** attempt)  # 指数退避
            logger.warning(f"⚠️ 触发限流 (429)，{wait_time:.1f} 秒后重试...")
            if attempt < max_retries:
                time.sleep(wait_time)
            else:
                return f"❌ 限流错误 (429): 已达最大重试次数。建议降低调用频率。"

        except APIStatusError as e:
            if e.status_code == 403:
                return f"❌ 权限错误 (403): 可能未开通 {model} 模型或余额不足。"
            elif e.status_code >= 500:
                logger.error(f"💥 服务端错误 ({e.status_code}): {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    return f"❌ 服务端错误 ({e.status_code}): 请稍后再试。"
            else:
                return f"❌ 请求错误 ({e.status_code}): {e.message}"

        except (APITimeoutError, APIConnectionError) as e:
            logger.error(f"🌐 网络错误: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                return "❌ 网络超时：请检查网络连接或重试。"

        except Exception as e:
            logger.exception("🔥 未知错误")
            return f"❌ 未知错误: {str(e)}"

    return "❌ 调用失败：达到最大重试次数"