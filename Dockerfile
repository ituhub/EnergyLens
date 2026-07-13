# ── Stage 1: Build React frontend ──
FROM node:20-slim AS frontend

WORKDIR /app/dashboard
COPY dashboard/package*.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

# ── Stage 2: Python backend + static frontend ──
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code (flat structure — no energylens/ prefix)
COPY api/ api/
COPY ml/ ml/
COPY core/ core/
COPY connectors/ connectors/
COPY config/ config/
COPY pipeline/ pipeline/
COPY auto_refresh.py .

# Copy data and models
COPY data/ data/
COPY models/ models/

# Copy built React app from Stage 1
COPY --from=frontend /app/dashboard/dist /app/static

ENV PORT=8080
EXPOSE ${PORT}

CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
