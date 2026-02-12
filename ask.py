#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 问答 CLI 工具
支持命令行提问或交互模式
"""

import sys
import argparse
import logging
import json
import os
from typing import Dict, Any

# 配置日志：默认 WARNING，--debug 时为 INFO
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def safe_call_qa(question: str, provider: str, model: str = None) -> Dict[str, Any]:
    """
    安全调用结构化问答，捕获所有可能异常
    """
    try:
        # 动态导入，避免启动时依赖缺失
        from structured_qa import call_structured_qa
        result = call_structured_qa(question=question, provider=provider, model=model)
        
        # 如果 structured_qa 返回了 error 字段，也视为失败
        if isinstance(result, dict) and "error" in result:
            logger.warning(f"模型返回错误: {result['error']}")
            return {
                "error": "模型处理失败",
                "details": result.get("raw_output", result["error"])
            }
        
        return result
        
    except ImportError as e:
        logger.error(f"模块导入失败: {e}")
        return {"error": "缺少必要模块", "details": str(e)}
    
    except Exception as e:
        logger.exception("调用模型时发生未预期错误")
        return {"error": "系统内部错误", "details": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="🤖 AI 问答 CLI 工具 —— 输入问题，获取结构化答案",
        epilog="示例: python ask.py \"报销流程是什么？\" --provider qwen"
    )
    parser.add_argument("question", nargs="?", help="你要问的问题（留空进入交互模式）")
    parser.add_argument("--provider", default="qwen", choices=["qwen", "qianfan"], 
                        help="选择模型提供商 (默认: qwen)")
    parser.add_argument("--model", help="指定具体模型名称（如 qwen-max）")
    parser.add_argument("--debug", action="store_true", help="启用调试日志")

    args = parser.parse_args()

    # 启用调试日志
    if args.debug:
        logging.getLogger().setLevel(logging.INFO)
        logger.info("调试模式已开启")

    # 检查 .env 文件是否存在（关键！）
    if not os.path.exists(".env"):
        logger.warning("未找到 .env 文件，请确保已配置 API 密钥")
        print("⚠️  警告: 未检测到 .env 配置文件，请先创建并填入 API 密钥！", file=sys.stderr)

    # 无问题参数 → 进入交互模式
    if not args.question:
        print("🤖 AI 问答机器人（输入 'quit' 或 'exit' 退出）")
        print(f"   当前提供商: {args.provider} | 模型: {args.model or '默认'}\n")
        
        while True:
            try:
                q = input("❓ 你的问题: ").strip()
                if q.lower() in ("quit", "exit", "q", ""):
                    print("👋 再见！")
                    break
                if q:
                    result = safe_call_qa(q, args.provider, args.model)
                    print_answer(result)
                    print("-" * 50)
            except KeyboardInterrupt:
                print("\n👋 被用户中断，再见！")
                break
            except EOFError:
                print("\n👋 输入结束，再见！")
                break
        return

    # 单次提问模式
    logger.info(f"收到问题: {args.question}")
    logger.info(f"使用提供商: {args.provider}, 模型: {args.model or '默认'}")

    result = safe_call_qa(args.question, args.provider, args.model)
    
    # 输出结果（始终为 JSON，便于脚本调用）
    try:
        output = json.dumps(result, ensure_ascii=False, indent=2)
        print(output)
    except Exception as e:
        # 极端情况 fallback
        print(f'{{"error": "输出序列化失败", "details": "{str(e)}"}}')


def print_answer(result: Dict[str, Any]):
    """在交互模式下美化输出"""
    if "error" in result:
        print(f"❌ 错误: {result['error']}")
        if "details" in result:
            print(f"   详情: {result['details']}")
    elif "answer" in result:
        print(f"💡 答案: {result['answer']}")
        if "question_type" in result:
            print(f"   类型: {result['question_type']}")
    else:
        print("❓ 未知响应格式:")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()