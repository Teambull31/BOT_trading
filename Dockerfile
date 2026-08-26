# Image de production de l'agent de trading adaptatif.
#
# Choix : image slim + utilisateur non-root + healthcheck sur l'endpoint de
# supervision. Le conteneur n'embarque aucune cle : elles arrivent par variables
# d'environnement au demarrage.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --upgrade pip && pip install -e .

COPY config/ ./config/
COPY scripts/ ./scripts/

# Utilisateur non privilegie : un conteneur de trading compromis ne doit pas
# etre root sur son hote.
RUN useradd --create-home --shell /bin/bash trader \
    && mkdir -p /app/data /app/logs /app/artifacts \
    && chown -R trader:trader /app
USER trader

EXPOSE 9090 9091 9092

# Le healthcheck interroge le dashboard : un conteneur vivant mais fige est
# aussi dangereux qu'un conteneur mort.
HEALTHCHECK --interval=60s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:9092/health || exit 1

CMD ["python", "-m", "trader.main", "paper"]
