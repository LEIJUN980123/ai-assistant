# pdf_utils.py
import os
from pathlib import Path
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def pdf_to_text(pdf_path: str, output_path: str = None) -> str:
    """
    将 PDF 文件提取为纯文本
    
    Args:
        pdf_path (str): 输入 PDF 文件路径
        output_path (str, optional): 输出 .txt 文件路径。若为 None，则返回文本不保存
    
    Returns:
        str: 提取的文本内容
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
    
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("文件必须是 .pdf 格式")
    
    try:
        logger.info(f"正在提取 PDF: {pdf_path}")
        text = extract_text(pdf_path)
        
        # 清理文本：移除多余空白行，保留段落结构
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        cleaned_text = "\n".join(lines)
        
        # 保存到文件（可选）
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            logger.info(f"文本已保存至: {output_path}")
        
        return cleaned_text
        
    except PDFSyntaxError:
        raise ValueError(f"无效的 PDF 文件: {pdf_path}")
    except Exception as e:
        logger.error(f"提取失败: {e}")
        raise