FROM registry.cn-hangzhou.aliyuncs.com/library/python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制Python脚本
COPY predict.py train.py ./

# 安装Python依赖
RUN pip install --no-cache-dir ultralytics pillow opencv-python

# 复制Node.js应用
COPY package*.json ./
RUN npm install

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p public/uploads public/results datasets models

# 暴露端口
EXPOSE 6002

# 启动命令
CMD ["node", "server.js"]
