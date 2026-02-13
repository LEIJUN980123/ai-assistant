# rag_system.py
import os
import json
import logging
import torch
import re
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 导入你已有的 Qwen 客户端
from qwen_client import call_qwen

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    def __init__(self, document_path: str, chunk_size: int = 500):
        """
        初始化 RAG 系统（使用语义感知文本分块）
        
        Args:
            document_path (str): JSON 文档路径（来自 pdf_to_json.py）
            chunk_size (int): 每个文本块的目标最大字符数（默认 500）
        """
        self.document_path = Path(document_path)
        self.chunk_size = chunk_size
        self.chunks = []          # List[str]
        self.chunk_metadatas = [] # List[Dict]
        self.index = None         # FAISS index
        self.embedding_model = None
        
        self._load_documents()
        self._chunk_documents()
        self._build_vector_index()
    
    def _load_documents(self):
        """从 JSON 加载文档内容"""
        if not self.document_path.exists():
            raise FileNotFoundError(f"文档不存在: {self.document_path}")
        
        with open(self.document_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 支持两种格式：
        # 1. 来自 pdf_to_json.py 的 {"pages": [...]}
        # 2. 简单列表 ["text1", "text2"]
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
        """按中英文句末标点切分句子"""
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
        """将长文本按语义边界（段落/句子）切分为小块"""
        self.chunks = []
        self.chunk_metadatas = []

        for i, text in enumerate(self.raw_texts):
            if not text.strip():
                continue

            source = self.sources[i]
            current_chunk = ""
            
            # 按自然段落分割（双换行符）
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for para in paragraphs:
                if len(para) <= self.chunk_size:
                    # 短段落：尝试合并到当前块
                    if current_chunk and len(current_chunk) + len(para) + 2 <= self.chunk_size:
                        current_chunk += "\n\n" + para
                    else:
                        # 保存当前块
                        if current_chunk:
                            self.chunks.append(current_chunk)
                            self.chunk_metadatas.append({
                                "source": source,
                                "chunk_id": len(self.chunks) - 1,
                                "start_char": -1,
                                "end_char": -1
                            })
                        current_chunk = para
                else:
                    # 长段落：按句子切分
                    sentences = self._split_into_sentences(para)
                    for sent in sentences:
                        if current_chunk and len(current_chunk) + len(sent) + 1 <= self.chunk_size:
                            current_chunk += " " + sent
                        else:
                            if current_chunk:
                                self.chunks.append(current_chunk)
                                self.chunk_metadatas.append({
                                    "source": source,
                                    "chunk_id": len(self.chunks) - 1,
                                    "start_char": -1,
                                    "end_char": -1
                                })
                            current_chunk = sent
            
            # 添加最后一块
            if current_chunk:
                self.chunks.append(current_chunk)
                self.chunk_metadatas.append({
                    "source": source,
                    "chunk_id": len(self.chunks) - 1,
                    "start_char": -1,
                    "end_char": -1
                })

        logger.info(f"✅ 切分为 {len(self.chunks)} 个语义文本块")

    def _build_vector_index(self):
        """构建 FAISS 向量索引（带缓存和安全检查）"""
        if self.embedding_model is None:
            logger.info("正在加载嵌入模型...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.embedding_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device=device
            )
            logger.info(f"✅ 嵌入模型加载完成（设备: {device}）")
        else:
            logger.info("复用已加载的嵌入模型")

        if not self.chunks:
            logger.warning("⚠️ 无可用于构建索引的文本块，跳过索引构建")
            self.index = None
            return

        logger.info(f"正在为 {len(self.chunks)} 个文本块生成向量...")
        embeddings = self.embedding_model.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype('float32')

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        logger.info(f"✅ FAISS 索引构建完成 | 维度: {dimension} | 文本块数量: {len(self.chunks)}")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """检索最相关的 k 个文本块"""
        if self.index is None:
            raise RuntimeError("向量索引未初始化")
        
        query_vector = self.embedding_model.encode([query])
        query_vector = np.array(query_vector).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "content": self.chunks[idx],
                    "metadata": self.chunk_metadatas[idx],
                    "score": float(distances[0][i])
                })
        return results
    
    def generate_answer(self, question: str, context_list: List[str], max_contexts: int = 3) -> str:
        """使用 Qwen 生成答案（基于检索到的上下文）"""
        if not context_list:
            return "根据现有资料无法确定"

        selected_contexts = []
        total_len = 0
        max_chars = 2000  # 保守限制，避免超出模型上下文

        for ctx in context_list[:max_contexts]:
            if total_len + len(ctx) > max_chars:
                break
            selected_contexts.append(ctx)
            total_len += len(ctx)

        context = "\n\n".join(selected_contexts)
        
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
            response = call_qwen(prompt, model="qwen-max")
            return response.strip()
        except Exception as e:
            logger.error(f"Qwen 调用失败: {e}")
            return "抱歉，生成答案时出现错误，请稍后重试。"
    
    def ask(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """端到端问答"""
        logger.info(f"🔍 检索中: '{question}'")
        retrieved = self.retrieve(question, k=top_k)
        contexts = [item["content"] for item in retrieved]
        answer = self.generate_answer(question, contexts)
        
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved
        }


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    DOC_PATH = "data/processed/Docker.json"
    
    rag = RAGSystem(DOC_PATH, chunk_size=500)
    
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
        
        print("\n📚 检索到的片段:")
        for i, chunk in enumerate(result["retrieved_chunks"][:2]):
            source = chunk["metadata"]["source"]
            preview = chunk["content"][:100].replace('\n', ' ')
            print(f"  [{i+1}] ({source}) {preview}...")