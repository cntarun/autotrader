# CPU-only image for the Kronos autotrader. Build from the repo root:
#   docker build -f deploy/Dockerfile -t kronos-autotrader:local .
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
COPY requirements-autotrader.txt .
RUN pip install -r requirements-autotrader.txt

# Bake the pinned model weights into the image: no runtime dependency on Hugging Face,
# and the exact model you backtested is the one that trades.
ARG TOKENIZER=NeoQuasar/Kronos-Tokenizer-base
ARG TOKENIZER_REV=0e0117387f39004a9016484a186a908917e22426
ARG PREDICTOR=NeoQuasar/Kronos-small
ARG PREDICTOR_REV=901c26c1332695a2a8f243eb2f37243a37bea320
RUN python -c "from huggingface_hub import snapshot_download as d; \
d('${TOKENIZER}', revision='${TOKENIZER_REV}'); d('${PREDICTOR}', revision='${PREDICTOR_REV}')"
ENV HF_HUB_OFFLINE=1

COPY model/ model/
COPY autotrader/ autotrader/

RUN useradd --uid 10001 --create-home trader && mkdir -p /state && chown trader /state
USER trader
ENV KRONOS_STATE_DIR=/state
VOLUME ["/state"]
EXPOSE 9108

HEALTHCHECK --interval=60s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9108/healthz',timeout=4).status==200 else 1)" || exit 1

ENTRYPOINT ["python", "-m", "autotrader.run"]
CMD ["--config", "/config/config.yaml", "--loop"]
