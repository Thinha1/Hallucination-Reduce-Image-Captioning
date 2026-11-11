# syntax=docker/dockerfile:1.7
# Một ảnh chạy cả backend (FastAPI, cổng 8000) lẫn frontend (Next.js, cổng 3000).
#
# Build (từ thư mục webapp/, nơi chứa file này):
#   docker build -t klcn-webapp .
# Chạy (mount thư mục mô hình, KHÔNG đóng gói vào ảnh vì ~2,8 GB):
#   docker run --rm --gpus all -p 3000:3000 -p 8000:8000 ^
#     -v "%cd%\backend\weights:/app/webapp/backend/weights:ro" klcn-webapp
# Hoặc dùng docker-compose.yml cùng thư mục: docker compose up --build
# Mở http://localhost:3000. API docs: http://localhost:8000/docs
#
# Trình duyệt gọi backend trực tiếp nên NEXT_PUBLIC_API_BASE_URL được nhúng lúc build. Mặc định
# http://localhost:8000 (chạy trên chính máy này). Nếu truy cập từ máy khác, build lại:
#   docker build --build-arg NEXT_PUBLIC_API_BASE_URL=http://<ip-may-chu>:8000 -t klcn-webapp .
#
# GPU: wheel torch cuXXX đã kèm sẵn thư viện CUDA, chỉ cần driver NVIDIA + nvidia-container-toolkit
# trên máy chủ. Không có GPU thì bỏ --gpus all, backend tự chạy CPU (chậm).
# Đổi bản CUDA: --build-arg TORCH_CUDA_TAG=cu124

ARG NODE_VERSION=22
ARG TORCH_CUDA_TAG=cu121

# ---------- 1) build frontend ----------
FROM node:${NODE_VERSION}-bookworm-slim AS frontend-build
WORKDIR /build
ENV NEXT_TELEMETRY_DISABLED=1
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
RUN npm run build && npm prune --omit=dev

# ---------- 2) ảnh chạy: Python (backend) + Node (frontend) ----------
FROM python:3.11-slim-bookworm AS app

ARG TORCH_CUDA_TAG

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Node chỉ cần để chạy `next start`
COPY --from=node:22-bookworm-slim /usr/local/bin/node /usr/local/bin/node

WORKDIR /app

# Python: cài torch bản CUDA trước, sau đó mới cài phần còn lại để pip không kéo bản CPU
# Ghim transformers 4.46.1 và ultralytics 8.4.34, đúng bản của môi trường Python 3.10 mà webapp
# đã chạy được. transformers 5.x đổi kiểu trả về của CLIP (backend/pipeline/retrieval.py không
# xử lý) và bản mới hơn còn đòi torch >= 2.6, trong khi torch cu121 chỉ có tới 2.5.1.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG}

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
        "transformers==4.46.1" "ultralytics==8.4.34" \
    && python -m nltk.downloader -d /usr/share/nltk_data wordnet omw-1.4 punkt punkt_tab

# Backend phải nằm đúng gói webapp.backend (code dùng import tương đối)
COPY __init__.py /app/webapp/__init__.py
COPY backend/ /app/webapp/backend/

# Frontend đã build sẵn
COPY --from=frontend-build /build /app/frontend

COPY <<'EOF' /app/start.sh
#!/bin/bash
# Chạy backend nền, frontend tiền cảnh. Một trong hai dừng thì container dừng.
cd /app
python -m uvicorn webapp.backend.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
cd /app/frontend
node node_modules/next/dist/bin/next start -H 0.0.0.0 -p 3000 &
WEB_PID=$!
trap 'kill $API_PID $WEB_PID 2>/dev/null' TERM INT
wait -n $API_PID $WEB_PID
kill $API_PID $WEB_PID 2>/dev/null
wait
EOF
RUN chmod +x /app/start.sh

ENV WEIGHTS_DIR=/app/webapp/backend/weights \
    PYTHONUNBUFFERED=1 \
    NEXT_TELEMETRY_DISABLED=1

VOLUME ["/app/webapp/backend/weights", "/app/webapp/backend/uploads"]
EXPOSE 3000 8000

# Backend nạp mô hình nặng lúc khởi động nên cho thời gian chờ dài
HEALTHCHECK --interval=30s --timeout=5s --start-period=240s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)" || exit 1

CMD ["/app/start.sh"]
