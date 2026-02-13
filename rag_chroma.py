# rag_chroma.py
import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb

# 自定义模块（复用 Day 10 的逻辑）
from embedding_client import get_embeddings  # 支持 MiniLM / Qwen
from qwen_client import call_qwen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGWithChroma:
    def __init__(
        self,
        document_path: str,
        chunk_size: int = 500,
        base_persist_dir: Optional[str] = "./chroma_db"
    ):
        """
        使用 Chroma 作为向量数据库的 RAG 系统
        
        Args:
            document_path: JSON 文档路径
            chunk_size: 文本分块大小
            base_persist_dir: 基础持久化目录（会自动追加 embedding 类型子目录）
        """
        self.document_path = Path(document_path)
        self.chunk_size = chunk_size
        
        # 🔥 自动根据 embedding 模式选择子目录
        use_qwen = os.getenv("USE_QWEN_EMBEDDING", "false").lower() == "true"
        embed_suffix = "qwen_1536" if use_qwen else "minilm_384"
        
        # 构建带文档名的路径，避免不同文档混用
        doc_name = self.document_path.stem  # e.g., "Docker"
        persist_directory = Path(base_persist_dir) / doc_name / embed_suffix if base_persist_dir else None

        if persist_directory:
            persist_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 使用持久化目录: {persist_directory}")
            self.client = chromadb.PersistentClient(path=str(persist_directory))
        else:
            logger.info("🧠 使用内存模式（不持久化）")
            self.client = chromadb.EphemeralClient()

        # 创建集合（自动跳过已存在）
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        # 加载并构建数据库
        self._load_documents()
        self._chunk_documents()
        self._build_chroma_index()

    def _load_documents(self):
        if not self.document_path.exists():
            raise FileNotFoundError(f"文档不存在: {self.document_path}")
        
        with open(self.document_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if "pages" in data:
            texts = [page["content"] for page in data["pages"]]
            sources = [f"{data['metadata']['source_file']}:p{page['page_number']}" 
                      for page in data["pages"]]
        else:
            texts = data if isinstance(data, list) else [data.get("text", "")]
            sources = ["unknown"] * len(texts)
        
        self.raw_texts = texts
        self.sources = sources
        logger.info(f"✅ 加载 {len(texts)} 段原始文本")

    def _split_into_sentences(self, text: str) -> List[str]:
        sentence_endings = r'[。！？\.!?]'
        parts = re.split(f'({sentence_endings})', text)
        sentences = []
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and re.match(sentence_endings, parts[i + 1]):
                sentences.append(parts[i] + parts[i + 1])
                i += 2
            else:
                if parts[i].strip():
                    sentences.append(parts[i])
                i += 1
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_documents(self):
        self.chunks = []
        self.metadatas = []
        self.ids = []

        chunk_id = 0
        for i, text in enumerate(self.raw_texts):
            if not text.strip():
                continue

            source = self.sources[i]
            current_chunk = ""
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for para in paragraphs:
                if len(para) <= self.chunk_size:
                    if current_chunk and len(current_chunk) + len(para) + 2 <= self.chunk_size:
                        current_chunk += "\n\n" + para
                    else:
                        if current_chunk:
                            self.chunks.append(current_chunk)
                            self.metadatas.append({"source": source})
                            self.ids.append(f"chunk_{chunk_id}")
                            chunk_id += 1
                        current_chunk = para
                else:
                    sentences = self._split_into_sentences(para)
                    for sent in sentences:
                        if current_chunk and len(current_chunk) + len(sent) + 1 <= self.chunk_size:
                            current_chunk += " " + sent
                        else:
                            if current_chunk:
                                self.chunks.append(current_chunk)
                                self.metadatas.append({"source": source})
                                self.ids.append(f"chunk_{chunk_id}")
                                chunk_id += 1
                            current_chunk = sent
            
            if current_chunk:
                self.chunks.append(current_chunk)
                self.metadatas.append({"source": source})
                self.ids.append(f"chunk_{chunk_id}")
                chunk_id += 1

        logger.info(f"✅ 切分为 {len(self.chunks)} 个语义文本块")

    def _build_chroma_index(self):
        """将文本块 + 向量 + 元数据存入 Chroma"""
        if not self.chunks:
            logger.warning("⚠️ 无文本块，跳过索引构建")
            return

        # 🌟 关键：使用我们自己的 embedding 函数（MiniLM 或 Qwen）
        logger.info("正在生成向量并存入 Chroma...")
        embeddings = get_embeddings(self.chunks).tolist()  # Chroma 需要 list[float]

        # 批量添加（Chroma 支持 up to 4168 条/次）
        batch_size = 1000
        for i in range(0, len(self.chunks), batch_size):
            self.collection.add(
                ids=self.ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=self.chunks[i:i+batch_size],
                metadatas=self.metadatas[i:i+batch_size]
            )

        logger.info(f"✅ Chroma 数据库构建完成 | 文本块数量: {len(self.chunks)}")

    def retrieve(
        self,
        query: str,
        k: int = 3,
        where: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        从 Chroma 检索相似文本
        
        Args:
            query: 查询问题
            k: 返回 top-k 结果
            where: 元数据过滤，如 {"source": "xxx.pdf:p5"}
        """
        query_embedding = get_embeddings([query]).tolist()[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # 转换为统一格式
        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i]  # Chroma cosine distance → similarity
            })
        return retrieved

    def generate_answer(self, question: str, context_list: List[str], max_contexts: int = 3) -> str:
        if not context_list:
            return "根据现有资料无法确定"

        selected = []
        total_len = 0
        max_chars = 2000
        for ctx in context_list[:max_contexts]:
            if total_len + len(ctx) > max_chars:
                break
            selected.append(ctx)
            total_len += len(ctx)

        context = "\n\n".join(selected)
        prompt = f"""你是一个严谨的 AI 助手，请严格根据以下【上下文】回答问题。
- 如果上下文包含足够信息，请直接给出**简洁、准确**的答案。
- 如果上下文不包含相关信息，请回答：“根据现有资料无法确定”。
- 不要编造信息，不要解释推理过程，不要添加额外说明。

上下文：
{context}

问题：
{question}

【回答】
"""
        try:
            return call_qwen(prompt, model="qwen-max").strip()
        except Exception as e:
            logger.error(f"Qwen 调用失败: {e}")
            return "生成答案时出错。"

    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        logger.info(f"🔍 Chroma 检索中: '{question}'")
        retrieved = self.retrieve(question, k=top_k)
        contexts = [item["content"] for item in retrieved]
        answer = self.generate_answer(question, contexts)
        return {"question": question, "answer": answer, "retrieved_chunks": retrieved}


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    import os
    # 切换 Embedding 模式
    os.environ["USE_QWEN_EMBEDDING"] = "true"  # ← 改这里！true=Qwen(1536d), false=MiniLM(384d)

    # 自动按文档名 + embedding 类型隔离数据库
    rag = RAGWithChroma(
        document_path="data/processed/Docker.json",
        chunk_size=500,
        base_persist_dir="./chroma_db"  # 最终路径: ./chroma_db/Docker/qwen_1536/
    )
    
    questions = [
        "Docker镜像如何构建？",
        "容器和镜像的区别是什么？",
        "如何查看运行中的容器？"
    ]
    
    for q in questions:
        print("\n" + "="*60)
        result = rag.ask(q)
        print(f"❓ 问题: {result['question']}")
        print(f"✅ 答案: {result['answer']}")
        
        print("\n📚 检索来源:")
        for chunk in result["retrieved_chunks"]:
            print(f"  - ({chunk['metadata']['source']})")