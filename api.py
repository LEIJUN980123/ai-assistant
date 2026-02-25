# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
from typing import Optional
import logging
import os
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



# ==============================
# 安全导入 RAG（防止因初始化失败导致 app 无法加载）
# ==============================
rag_system = None
try:
    from rag_langchain import LangChainRAGWithMemory

    # 确保文档路径正确
    BASE_DIR = Path(__file__).parent.resolve()
    DOCUMENT_PATH = BASE_DIR / "data" / "processed" / "Docker.json"

    if not DOCUMENT_PATH.exists():
        raise FileNotFoundError(f"文档不存在: {DOCUMENT_PATH}")

    logger.info("🚀 正在加载 RAG 系统...")
    logger.info(f"📄 使用文档: {DOCUMENT_PATH}")

    rag_system = LangChainRAGWithMemory(
        document_path=str(DOCUMENT_PATH),
        chunk_size=500,
        chunk_overlap=50
    )
    logger.info("✅ RAG 系统加载完成")

except Exception as e:
    logger.critical(f"❌ RAG 系统初始化失败: {e}", exc_info=True)
    # 不退出，允许 app 启动（便于查看 /health 和错误信息）

# ==============================
# 1. 初始化 FastAPI 应用（必须在顶层！）
# ==============================
app = FastAPI(
    title="RAG 问答系统 API",
    description="基于 LangChain + Qwen 的 Docker 技术问答助手，支持多轮对话",
    version="1.0.0"
)
# ✅ 必须放在最前面！
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法（包括 OPTIONS）
    allow_headers=["*"],  # 允许所有头
)

# ==============================
# 2. 定义请求/响应模型
# ==============================
class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class AskResponse(BaseModel):
    answer: str
    session_id: str

# ==============================
# 3. 定义 API 路由
# ==============================
@app.post("/ask", response_model=AskResponse, summary="提问接口")
async def ask_question(request: AskRequest):
    """
    向 RAG 系统提问：
    - 如果提供 session_id，将启用多轮对话记忆
    - 返回结构化 JSON 答案
    """
    if rag_system is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 系统未就绪，请检查服务日志"
        )
    
    try:
        answer = rag_system.ask(request.question, session_id=request.session_id)
        return AskResponse(answer=answer, session_id=request.session_id)
    except Exception as e:
        logger.error(f"处理问题时出错: {e}")
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

@app.get("/health", summary="健康检查")
async def health_check():
    status = "ok" if rag_system is not None else "degraded"
    message = "RAG API is running" if rag_system else "RAG not loaded"
    return {
        "status": status,
        "message": message,
        "rag_ready": rag_system is not None
    }

# ==============================
# 4. 启动入口（可选）
# ==============================
if __name__ == "__main__":
    import uvicorn
    logger.info("🔧 启动 Uvicorn 开发服务器...")
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )