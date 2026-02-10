# test_qwen.py
from qwen_client import call_qwen

def test_normal():
    print("【正常调用】")
    ans = call_qwen("你好，请用一句话介绍你自己。")
    print("回答:", ans)
    print("-" * 50)

def test_invalid_key():
    print("【模拟无效 Key】")
    import os
    old_key = os.environ.get("DASHSCOPE_API_KEY")
    os.environ["DASHSCOPE_API_KEY"] = "invalid-key"
    
    ans = call_qwen("测试")
    print("结果:", ans)
    
    # 恢复
    if old_key:
        os.environ["DASHSCOPE_API_KEY"] = old_key
    print("-" * 50)

if __name__ == "__main__":
    #test_normal()
     test_invalid_key()  # 取消注释可测试错误处理