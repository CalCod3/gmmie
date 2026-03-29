# ---- Base Image ----
FROM python:3.11-slim@sha256:latest

# ---- Environment ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- System Dependencies ----
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Work Directory ----
WORKDIR /app

# ---- Install Dependencies ----
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- Copy Project Files ----
COPY . .

# ---- Expose Port (must match config / delivery) ----
EXPOSE 8001

# ---- Healthcheck ----
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# ---- Start FULL SYSTEM (engine + API) ----
CMD ["python", "main.py"]