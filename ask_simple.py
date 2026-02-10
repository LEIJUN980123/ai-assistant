# ask_simple.py
from llm_client import call_llm

question = "用一句话解释人工智能。"

print("=" * 50)
print("【1】通义千问:")
ans1 = call_llm("qwen", question)
print(ans1)

print("\n" + "=" * 50)
print("【2】百度千帆:")
ans2 = call_llm("qianfan", question, model="ernie-3.5-8k")
print(ans2)