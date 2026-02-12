#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 PDF 文档转换为结构化 JSON 格式，适用于 RAG 系统。
输出包含每页内容、页码、文件名等元信息。

使用示例:
    python pdf_to_json.py input.pdf output.json
    python pdf_to_json.py data/policy.pdf data/processed/policy.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 尝试导入 pdfminer.six
try:
    from pdfminer.pdfpage import PDFPage
    from pdfminer.layout import LAParams
    from pdfminer.converter import TextConverter
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from io import StringIO
except ImportError:
    print("❌ 错误: 未安装 pdfminer.six，请运行: pip install pdfminer.six")
    sys.exit(1)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_pages_from_pdf(pdf_path: str) -> List[str]:
    """
    从 PDF 中按页面提取原始文本
    
    Args:
        pdf_path (str): PDF 文件路径
    
    Returns:
        List[str]: 每个元素对应一页的文本内容
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")
    
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("输入文件必须是 .pdf 格式")
    
    pages = []
    try:
        with open(pdf_path, "rb") as fp:
            # 获取所有页面
            #pages_list = list(PDFPage.get_pages(fp, caching=True, check_extractable=True))
            pages_list = list(PDFPage.get_pages(fp, caching=True, check_extractable=False))
            for page in pages_list:
                resource_manager = PDFResourceManager()
                fake_file_handle = StringIO()
                # 使用 LAParams 控制布局解析参数（如合并行）
                converter = TextConverter(
                    resource_manager, 
                    fake_file_handle, 
                    laparams=LAParams(
                        char_margin=2.0,    # 字符间距容忍度
                        word_margin=0.1,    # 单词间距容忍度
                        line_margin=0.5     # 行间距容忍度
                    )
                )
                page_interpreter = PDFPageInterpreter(resource_manager, converter)
                
                page_interpreter.process_page(page)
                text = fake_file_handle.getvalue()
                
                pages.append(text)
                converter.close()
                fake_file_handle.close()
                
    except Exception as e:
        if "PDFSyntaxError" in str(type(e)):
            raise ValueError(f"无效或损坏的 PDF 文件: {pdf_path}")
        else:
            raise RuntimeError(f"PDF 解析失败: {e}")
    
    return pages


def clean_text(text: str) -> str:
    """
    清理文本：移除多余空白行，保留段落结构
    """
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:  # 跳过空行
            lines.append(stripped)
    return "\n".join(lines)


def pdf_to_structured_json(pdf_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    将 PDF 转换为结构化 JSON
    
    Args:
        pdf_path (str): 输入 PDF 路径
        output_path (str, optional): 输出 JSON 路径
    
    Returns:
        dict: 结构化数据
    """
    logger.info(f"开始处理 PDF: {pdf_path}")
    
    # 提取原始页面
    raw_pages = extract_pages_from_pdf(pdf_path)
    logger.info(f"成功提取 {len(raw_pages)} 页")
    
    # 清理每页文本
    cleaned_pages = [clean_text(page) for page in raw_pages]
    
    # 构建结果结构
    result = {
        "metadata": {
            "source_file": os.path.basename(pdf_path),
            "absolute_path": os.path.abspath(pdf_path),
            "total_pages": len(cleaned_pages),
            "conversion_tool": "pdfminer.six",
            "note": "适用于 RAG 系统的结构化文档"
        },
        "pages": [
            {
                "page_number": i + 1,
                "content": content
            }
            for i, content in enumerate(cleaned_pages)
        ]
    }
    
    # 保存到文件（如果指定了输出路径）
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ JSON 已保存至: {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="将 PDF 转换为结构化 JSON（RAG 友好格式）",
        epilog="示例: python pdf_to_json.py policy.pdf output.json"
    )
    parser.add_argument("input", help="输入 PDF 文件路径")
    parser.add_argument("output", nargs="?", help="输出 JSON 文件路径（可选）")
    parser.add_argument("--no-save", action="store_true", help="仅打印结果，不保存文件")
    
    args = parser.parse_args()
    
    try:
        if args.no_save:
            result = pdf_to_structured_json(args.input)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            output = args.output or f"{Path(args.input).stem}.json"
            pdf_to_structured_json(args.input, output)
            
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()