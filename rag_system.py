# rag_system.py
import os
import json
import logging
import torch
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
    def __init__(self, document_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化 RAG 系统
        
        Args:
            document_path (str): JSON 文档路径（来自 pdf_to_json.py）
            chunk_size (int): 每个文本块的最大字符数
            chunk_overlap (int): 块之间重叠字符数
        """
        self.document_path = Path(document_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
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
    
    def _chunk_documents(self):
        """将长文本切分为小块"""
        self.chunks = []
        self.chunk_metadatas = []
        for i, text in enumerate(self.raw_texts):
            if not text.strip():
                continue
            text_len=len(text)
            start = 0
            while start < text_len:
                end = start + self.chunk_size
                
                chunk = text[start:end]
                if chunk.strip():
                    self.chunks.append(chunk)
                    self.chunk_metadatas.append({
                        "source": self.sources[i],
                        "chunk_id": len(self.chunks) - 1,
                        "start_char": start,
                        "end_char": min(end, text_len)
                    })
                
                if end >= text_len:
                    break  # 到达末尾，退出
        
                # 计算下一块的起始位置
                start = end - self.chunk_overlap
        
                # 安全检查：防止不前进
                if start >= end:
                    start = end
                
        logger.info(f"✅ 切分为 {len(self.chunks)} 个文本块")
    
    def _build_vector_index(self):
        """构建 FAISS 向量索引（带缓存和安全检查）"""
    
        # === 1. 懒加载嵌入模型 ===
        if self.embedding_model is None:
            logger.info("正在加载嵌入模型...")
            self.embedding_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            logger.info("✅ 嵌入模型加载完成")
        else:
            logger.info("复用已加载的嵌入模型")

        # === 2. 安全检查：是否有文本块 ===
        if not self.chunks:
            logger.warning("⚠️ 无可用于构建索引的文本块，跳过索引构建")
            self.index = None
            return

        # === 3. 生成向量 ===
        logger.info(f"正在为 {len(self.chunks)} 个文本块生成向量...")
        embeddings = self.embedding_model.encode(
            self.chunks,
            show_progress_bar=True,
            convert_to_numpy=True  # 显式指定返回 numpy array
        ).astype('float32')

        # === 4. 构建 FAISS 索引 ===
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        logger.info(f"✅ FAISS 索引构建完成 | 维度: {dimension} | 文本块数量: {len(self.chunks)}")
    
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """检索最相关的 k 个文本块"""
        if self.index is None:
            raise RuntimeError("向量索引未初始化")
        
        # 生成查询向量
        query_vector = self.embedding_model.encode([query])
        query_vector = np.array(query_vector).astype('float32')
        
        # 检索
        distances, indices = self.index.search(query_vector, k)
        
        # 构建结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "content": self.chunks[idx],
                    "metadata": self.chunk_metadatas[idx],
                    "score": float(distances[0][i])
                })
        
        return results
    
    def generate_answer(self, question: str, context_list: List[str]) -> str:
        """使用 Qwen 生成答案"""
        # 拼接上下文（限制长度）
        context = "\n\n".join(context_list[:2])  # 只用前2个最相关
        
        prompt = f"""你是一个专业助手，请基于以下上下文回答问题。
如果上下文不包含答案，请回答“根据现有资料无法确定”。

上下文：
{context}

问题：{question}

请直接给出简洁答案，不要解释过程。"""
        
        return call_qwen(prompt, model="qwen-max")
    
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
    # 配置路径（根据你的实际路径修改）
    DOC_PATH = "data/processed/Docker.json"
    
    # 初始化 RAG 系统
    rag = RAGSystem(DOC_PATH)
    
    # 测试问题
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
        
        # 打印来源（调试用）
        print("\n📚 检索到的片段:")
        for i, chunk in enumerate(result["retrieved_chunks"][:2]):
            source = chunk["metadata"]["source"]
            preview = chunk["content"][:100].replace('\n', ' ')
            print(f"  [{i+1}] ({source}) {preview}...")