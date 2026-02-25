# rag_langchain.py
import os
from pathlib import Path
from typing import List, Optional
from collections import defaultdict, deque
import json
import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage
from langchain_chroma import Chroma

# 自定义模块（确保这些文件存在）
from embedding_client import get_embeddings
from qwen_client import call_qwen

# DuckDuckGo 搜索
from duckduckgo_search import DDGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局会话历史（轻量级 demo 用）
SESSION_HISTORY = defaultdict(lambda: deque(maxlen=2))


# ==============================
# 1. 自定义 Embedding Function
# ==============================
class CustomEmbeddingFunction:
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = get_embeddings(input)
        return embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self(texts)

    def embed_query(self, text: str) -> List[float]:
        emb = self([text])
        vec = emb[0]
        if hasattr(vec, 'tolist'):
            return vec.tolist()
        return list(vec)


# ==============================
# 2. 自定义 LLM（Qwen）
# ==============================
class CustomQwenLLM:
    def invoke(self, input: str | BaseMessage) -> str:
        if hasattr(input, 'content'):
            prompt = input.content
        else:
            prompt = str(input)
        return call_qwen(prompt, model="qwen-max")

    def __call__(self, input: str | BaseMessage, config=None) -> str:
        return self.invoke(input)


# ==============================
# 3. 可信搜索函数（DuckDuckGo）
# ==============================
def trusted_search(query: str, num_results: int = 3) -> str:
    """使用 DuckDuckGo 免费搜索获取摘要"""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                keywords=query,
                region="zh-cn",
                safesearch="moderate",
                max_results=num_results
            )
            snippets = [r["body"] for r in results if r.get("body")]
            return "\n".join(snippets[:num_results])
    except Exception as e:
        logger.warning(f"🔍 DuckDuckGo 搜索失败: {e}")
        return ""


# ==============================
# 4. 主 RAG 类
# ==============================
class LangChainRAG:
    def __init__(
        self,
        document_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        base_persist_dir: str = "./chroma_db_lc"
    ):
        self.document_path = Path(document_path)
        use_qwen = os.getenv("USE_QWEN_EMBEDDING", "false").lower() == "true"
        embed_suffix = "qwen_1536" if use_qwen else "minilm_384"
        doc_name = self.document_path.stem
        persist_directory = Path(base_persist_dir) / doc_name / embed_suffix
        persist_directory.mkdir(parents=True, exist_ok=True)

        documents = self._load_documents()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        embedding_func = CustomEmbeddingFunction()
        self.vectorstore = Chroma(
            collection_name="rag_docs",
            embedding_function=embedding_func,
            persist_directory=str(persist_directory),
            collection_metadata={"hnsw:space": "cosine"}
        )

        if self.vectorstore._collection.count() == 0:
            logger.info(f"🔄 首次构建向量库，共 {len(splits)} 个文本块")
            self.vectorstore.add_documents(splits)
        else:
            logger.info(f"📂 加载已有向量库，共 {self.vectorstore._collection.count()} 个文本块")

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.qa_chain = self._build_rag_chain()

    def _load_documents(self) -> List[Document]:
        with open(self.document_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        if "pages" in data:
            for page in data["pages"]:
                content = page["content"]
                source = f"{data['metadata']['source_file']}:p{page['page_number']}"
                docs.append(Document(page_content=content, metadata={"source": source}))
        else:
            texts = data if isinstance(data, list) else [data.get("text", "")]
            for text in texts:
                docs.append(Document(page_content=text, metadata={"source": "unknown"}))
        return docs

    def _build_rag_chain(self):
        prompt_template = """你是一个严谨的 AI 助手，请严格根据以下【上下文】回答问题。
- 如果上下文包含足够信息，请直接给出**简洁、准确**的答案。
- 如果上下文不包含相关信息，请回答：“根据现有资料无法确定”。
- 不要编造信息，不要解释推理过程，不要添加额外说明。

上下文：
{context}

问题：
{question}

【回答】
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = CustomQwenLLM()

        def format_docs(docs):
            total_len = 0
            selected = []
            max_chars = 2000
            for d in docs:
                if total_len + len(d.page_content) > max_chars:
                    break
                selected.append(d.page_content)
                total_len += len(d.page_content)
            return "\n\n".join(selected)

        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        return rag_chain

    def ask(self, question: str) -> str:
        return self.qa_chain.invoke(question)


# ==============================
# 5. 带记忆 + 搜索兜底的 RAG
# ==============================
class LangChainRAGWithMemory(LangChainRAG):
    def ask(self, question: str, session_id: Optional[str] = None) -> str:
        # Step 1: 构建实际提问内容（带或不带历史）
        if session_id is None:
            actual_question = question
            is_with_memory = False
        else:
            history = SESSION_HISTORY[session_id]
            if history:
                history_text = "\n".join([
                    f"用户之前问：{q}\n助手回答：{a}"
                    for q, a in history
                ])
                actual_question = (
                    f"【对话历史】\n{history_text}\n\n"
                    f"【当前问题】\n{question}"
                )
            else:
                actual_question = question
            is_with_memory = True

        # Step 2: 先走本地 RAG
        answer = super().ask(actual_question)

        # Step 3: 如果无答案，触发搜索
        if "根据现有资料无法确定" in answer:
            logger.info("🔍 本地无答案，触发 DuckDuckGo 搜索...")
            search_results = trusted_search(question)  # 用原始 question 搜索
            if search_results.strip():
                fallback_prompt = f"""你是一个严谨的 AI 助手，请基于以下【网络搜索结果】回答问题。
- 只使用搜索结果中的信息，不要编造。
- 如果结果不相关或为空，请回答“未找到相关信息”。

【搜索结果】
{search_results}

【问题】
{question}

【回答】
"""
                llm = CustomQwenLLM()
                answer = llm.invoke(fallback_prompt)
            else:
                answer = "未找到相关信息"

        # Step 4: 保存到会话历史（仅当 session_id 提供时）
        if session_id is not None:
            SESSION_HISTORY[session_id].append((question, answer))

        return answer


