# structured_qa.py
"""
Day 4: Prompt Engineering 实战
目标：让 LLM 返回 {"question_type": "...", "answer": "..."} 的合法 JSON
"""

import json
import re
from typing import Dict, Any
from qwen_client import call_qwen  # 复用 Day 3 的客户端

def build_structured_prompt(user_question: str) -> str:
    """构建强约束的 Prompt"""
    return f"""
你是一个专业的问答系统，请严格按以下规则回答：

1. **角色**：你是一个精准、简洁的 AI 助手。
2. **任务**：分析用户问题，返回一个 JSON 对象，包含：
   - "question_type": 问题类型（如 "fact", "opinion", "calculation", "weather", "other"）
   - "answer": 针对问题的直接答案（不超过 100 字）
3. **格式要求**：
   - 必须是纯 JSON，无任何额外文字、注释、Markdown
   - JSON 必须能被 Python json.loads() 解析
   - 不要包含 ```json 或 ``` 等包裹符

示例：
用户问题：地球的半径是多少？
输出：{{"question_type": "fact", "answer": "地球平均半径约为6371公里。"}}

用户问题：你觉得人工智能会取代人类吗？
输出：{{"question_type": "opinion", "answer": "这是一个有争议的话题，取决于具体领域和伦理框架。"}}

现在，请处理以下问题：
用户问题：{user_question}
输出：
""".strip()

def parse_llm_response(raw_output: str) -> Dict[str, str]:
    """
    尝试从模型输出中提取合法 JSON
    支持：纯 JSON / 包裹在 ```json 中 / 带多余文本
    """
    # 方法 1：直接解析
    try:
        return json.loads(raw_output.strip())
    except json.JSONDecodeError:
        pass

    # 方法 2：提取 ```json ... ``` 中的内容
    json_match = re.search(r"```(?:json)?\s*({.*?})\s*```", raw_output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 方法 3：查找第一个 { 和最后一个 }
    brace_start = raw_output.find("{")
    brace_end = raw_output.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidate = raw_output[brace_start:brace_end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 失败
    raise ValueError(f"无法解析为 JSON: {raw_output[:100]}...")

def call_structured_qa(
    question: str,
    provider: str = "qwen",
    model: str = None,
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    调用 LLM 获取结构化问答结果
    
    Returns:
        成功: {"question_type": "...", "answer": "...", "source": "qwen"}
        失败: {"error": "原因", "raw_output": "..."}
    """
    prompt = build_structured_prompt(question)
    
    for attempt in range(max_retries + 1):
        raw_response = call_qwen(prompt)
        
        # 检查是否已是错误信息
        if raw_response.startswith("❌"):
            return {"error": raw_response, "raw_output": raw_response}
        
        try:
            parsed = parse_llm_response(raw_response)
            # 验证必要字段
            if "question_type" not in parsed or "answer" not in parsed:
                raise ValueError("缺少必要字段")
            return {
                "question_type": parsed["question_type"],
                "answer": parsed["answer"],
                "source": provider,
                "raw_output": raw_response  # 用于调试
            }
        except Exception as e:
            if attempt < max_retries:
                print(f"  → JSON 解析失败 (尝试 {attempt+1})，重试...")
                continue
            return {
                "error": f"JSON 解析失败: {str(e)}",
                "raw_output": raw_response
            }
    
    return {"error": "未知错误", "raw_output": ""}