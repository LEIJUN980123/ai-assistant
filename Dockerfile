# Dockerfile
FROM python:3.10-slim

# 设置时区（避免时间警告）
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 设置工作目录
WORKDIR /app

# 复制依赖文件（利用缓存加速构建）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有代码和资源
COPY . .

# 创建非 root 用户（安全最佳实践）
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["python", "rag_web_ui.py"]