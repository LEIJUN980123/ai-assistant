# test_structured.py
from structured_qa import call_structured_qa
import json

def test_cases():
    questions = [
        "巴黎是哪个国家的首都？",
        "2的10次方等于多少？",
        "推荐一部好看的科幻电影。",
        "今天上海的天气如何？"
    ]
    
    for q in questions:
        print(f"\n❓ 问题: {q}")
        print("-" * 50)
        
        # 测试 Qwen
        result = call_structured_qa(q)
        if "error" not in result:
            print("✅ Qwen 结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("❌ Qwen 错误:", result["error"])
        
        # 测试 Qianfan（可选）
        # result2 = call_structured_qa(q, provider="qianfan", model="ernie-3.5-8k")
        # ... 类似处理

if __name__ == "__main__":
    test_cases()