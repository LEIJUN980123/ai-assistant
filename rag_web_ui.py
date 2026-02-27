# rag_web_ui.py
import gradio as gr

# 导入 RAG 系统
try:
    from rag_langchain import LangChainRAGWithMemory as LangChainRAG
except ImportError:
    from rag_langchain import LangChainRAG

# 初始化 RAG（只加载一次）
rag_system = LangChainRAG(
    document_path="output/combined_docs.json",
    chunk_size=500,
    chunk_overlap=50
)

def predict(message: str, history: list) -> str:
    """返回纯字符串，ChatInterface 自动处理格式"""
    try:
        retrieved = rag_system.retriever.invoke(message)
        answer = rag_system.ask(message)

        sources = set()
        for doc in retrieved:
            source = doc.metadata.get("source", "未知来源")
            sources.add(source)
        
        source_text = "\n".join(f"• {s}" for s in sorted(sources)) if sources else "• 无明确来源"
        return f"{answer}"

    except Exception as e:
        return f"❌ 系统出错：{str(e)}"

# ✅ 正确方式：通过 chatbot 参数传递 avatar_images
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 隆祎SAP_MM模组知识问答助手")
    gr.Markdown("基于内部文档的智能问答系统，所有回答均来自提供的资料。")
    
    # 使用 ChatInterface 内嵌到 Blocks（可选），或直接用 ChatInterface
    chat_interface = gr.ChatInterface(
        fn=predict,

        title=None,  # 因为上面已用 Markdown 写标题
        description=None,
        # ⬇️ 关键：通过 chatbot 参数设置头像
        chatbot=gr.Chatbot(
            avatar_images=(
            "https://cdn-icons-png.flaticon.com/512/149/149071.png",  # 用户
            "https://cdn-icons-png.flaticon.com/512/4712/4712129.png"   # 机器人
            ),
            height=500
        ),
        textbox=gr.Textbox(
            placeholder="请输入您的问题...",
            container=False,
            scale=7
        )
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)