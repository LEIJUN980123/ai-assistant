# AI 问答 CLI 机器人

## 快速开始
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY

# 3. 提问
python ask.py "巴黎是哪个国家的首都？"