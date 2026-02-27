# rag_system.py
import json
import logging
import re
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any

# 自定义模块
from embedding_client import get_embeddings
from qwen_client import call_qwen  # 你已有的 Qwen 调用函数

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, document_path: str, chunk_size: int = 500):
        self.document_path = Path(document_path)
        self.chunk_size = chunk_size
        self.chunks = []
        self.chunk_metadatas = []
        self.index = None
        
        self._load_documents()
        self._chunk_documents()
        self._build_vector_index()

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
        self.chunk_metadatas = []

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
                            self.chunk_metadatas.append({"source": source})
                        current_chunk = para
                else:
                    sentences = self._split_into_sentences(para)
                    for sent in sentences:
                        if current_chunk and len(current_chunk) + len(sent) + 1 <= self.chunk_size:
                            current_chunk += " " + sent
                        else:
                            if current_chunk:
                                self.chunks.append(current_chunk)
                                self.chunk_metadatas.append({"source": source})
                            current_chunk = sent
            
            if current_chunk:
                self.chunks.append(current_chunk)
                self.chunk_metadatas.append({"source": source})

        logger.info(f"✅ 切分为 {len(self.chunks)} 个语义文本块")

    def _build_vector_index(self):
        if not self.chunks:
            logger.warning("⚠️ 无文本块，跳过索引构建")
            self.index = None
            return

        # 🌟 关键：自动选择 embedding 方式
        embeddings = get_embeddings(self.chunks)
        dimension = embeddings.shape[1]
        logger.info(f"✅ 生成 {len(self.chunks)} 个 {dimension} 维向量")

        self.index = faiss.IndexFlatIP(dimension)  # 使用内积（需 normalized）
        faiss.normalize_L2(embeddings)  # 确保向量归一化
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("向量索引未初始化")
        
        query_emb = get_embeddings([query])
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "content": self.chunks[idx],
                    "metadata": self.chunk_metadatas[idx],
                    "score": float(distances[0][i])  # 越接近 1 越相似
                })
        return results

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
        prompt =  f"""你是一个严谨的 AI 助手，请严格根据以下【上下文】回答问题。
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
        logger.info(f"🔍 检索中: '{question}'")
        retrieved = self.retrieve(question, k=top_k)
        contexts = [item["content"] for item in retrieved]
        answer = self.generate_answer(question, contexts)
        return {"question": question, "answer": answer, "retrieved_chunks": retrieved}

