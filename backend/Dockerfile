FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
RUN pip install .

COPY config.yaml ./

# Caches SQLite (embeddings, réponses LLM, reranking) dans /data : un volume avec docker compose,
# un disque éphémère chez l'hébergeur (vidé à chaque redémarrage, ce ne sont que des caches).
RUN useradd --create-home appuser && mkdir -p /data && chown appuser /data
USER appuser

# Port imposé par l'hébergeur (variable PORT, 10000 chez Render), 8000 sinon.
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
