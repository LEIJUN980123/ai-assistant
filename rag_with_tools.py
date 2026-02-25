# rag_with_tools.py
import os
from typing import List, Optional, Any
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnablePassthrough

# 假设你已有基础 RAG（用于文档问答）
from rag_langchain import LangChainRAG  # 你的原版（无记忆）

# ==============================
# 1. 定义 Tools（外部函数）
# ==============================

@tool
def get_weather(location: str) -> str:
    """查询指定城市的当前天气。参数：location（城市名，如'北京'）"""
    # 模拟 API 调用
    weather_map = {"北京": "晴，25°C", "上海": "多云，28°C", "广州": "雷阵雨，30°C"}
    return weather_map.get(location, f"未知城市：{location}")

@tool
def query_employee(name: str) -> str:
    """从员工数据库查询姓名对应的工号。参数：name（员工姓名）"""
    db = {"张三": "E1001", "李四": "E1002", "王五": "E1003"}
    return db.get(name, f"未找到员工：{name}")

tools = [get_weather, query_employee]

# ==============================
# 2. 构建支持 Tool Calling 的 LLM
# ==============================
llm = ChatOpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-max",
    temperature=0.0
)
llm_with_tools = llm.bind_tools(tools)

# ==============================
# 3. 构建混合系统：先尝试用工具，再 fallback 到 RAG
# ==============================
class ToolAugmentedRAG:
    def __init__(self, document_path: str):
        # 初始化你的原版 RAG（用于非工具类问题）
        self.rag = LangChainRAG(document_path)
        self.llm = llm
        self.llm_with_tools = llm_with_tools
        self.tools_by_name = {tool.name: tool for tool in tools}

    def _handle_tool_call(self, user_input: str) -> Optional[str]:
        """尝试用工具回答，若无工具调用则返回 None"""
        messages = [HumanMessage(content=user_input)]
        
        # 第一步：让 LLM 决定是否调用工具
        ai_msg = self.llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # 检查是否有工具调用
        if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_func = self.tools_by_name[tool_name]
                result = tool_func.invoke(tool_args)
                # 添加工具返回结果
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            
            # 第二步：让 LLM 生成最终答案
            final_response = self.llm.invoke(messages)
            return final_response.content
        else:
            return None  # 无工具调用，交由 RAG 处理

    def ask(self, question: str) -> str:
        # 先尝试工具
        tool_answer = self._handle_tool_call(question)
        if tool_answer is not None:
            return tool_answer
        # 否则 fallback 到你的 RAG
        return self.rag.ask(question)


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    os.environ["DASHSCOPE_API_KEY"] = "sk-1b243265bb7d4143b37badb65610b558"  # ← 替换为你的密钥
    os.environ["USE_QWEN_EMBEDDING"] = "true"

    system = ToolAugmentedRAG("data/processed/Docker.json")

    test_questions = [
        "今天北京天气怎么样？",           # → 调用 get_weather
        "张三的工号是多少？",             # → 调用 query_employee
        "Docker镜像如何构建？",          # → 走 RAG
        "怎么删除一个容器？",             # → 走 RAG
        "上海现在多少度？"               # → 调用 get_weather
    ]

    for q in test_questions:
        print("\n" + "=" * 60)
        print(f"❓ {q}")
        ans = system.ask(q)
        print(f"✅ {ans}")