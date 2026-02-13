# embedding_client.py
import os
import logging
import numpy as np
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ==============================
# 全局缓存变量（避免重复加载）
# ==============================
_LOCAL_EMBEDDING_MODEL = None
_DASHSCOPE_CONFIGURED = False


def _get_local_model():
    global _LOCAL_EMBEDDING_MODEL
    if _LOCAL_EMBEDDING_MODEL is None:
        logger.info("🔄 首次加载本地 MiniLM 模型（384 维）...")
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _LOCAL_EMBEDDING_MODEL = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2",
                device=device
            )
            logger.info(f"✅ MiniLM 模型加载完成（设备: {device}）")
        except Exception as e:
            raise RuntimeError(f"加载 MiniLM 失败: {e}")
    return _LOCAL_EMBEDDING_MODEL


def get_local_embeddings(texts: List[str]) -> np.ndarray:
    model = _get_local_model()  # 复用已加载模型
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    return embeddings


def get_qwen_embeddings(texts: List[str]) -> np.ndarray:
    global _DASHSCOPE_CONFIGURED
    try:
        import dashscope
        from dashscope import TextEmbedding
    except ImportError:
        raise ImportError("请安装: pip install dashscope")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("❌ 未设置 DASHSCOPE_API_KEY")

    if not _DASHSCOPE_CONFIGURED:
        dashscope.api_key = api_key
        _DASHSCOPE_CONFIGURED = True
        logger.info("☁️ Qwen Embedding 已配置（API Key 设置成功）")

    embeddings = []
    batch_size = 25
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = TextEmbedding.call(
            model="text-embedding-v2",
            input=batch
        )
        if response.status_code != 200:
            raise RuntimeError(f"Qwen API 错误: {response.code} - {response.message}")
        embeddings.extend([item["embedding"] for item in response.output["embeddings"]])
    
    return np.array(embeddings, dtype=np.float32)


def get_embeddings(texts: List[str]) -> np.ndarray:
    use_qwen = os.getenv("USE_QWEN_EMBEDDING", "false").lower() == "true"
    if use_qwen:
        return get_qwen_embeddings(texts)
    else:
        return get_local_embeddings(texts)