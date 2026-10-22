FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_DIR=/app/artifacts/model PORT=7860

WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY caption_bot/ caption_bot/
COPY app/ app/
COPY artifacts/model/ artifacts/model/

# Download the ImageNet CNN weights at build time so the first request is fast.
# Skipped automatically when the model ships its own fine tuned encoder weights.
RUN python -c "import pathlib, keras; from caption_bot.config import Config; \
from caption_bot.model import build_backbone; \
cfg = Config.load('artifacts/model/config.json'); \
fine_tuned = pathlib.Path('artifacts/model/encoder.weights.h5').exists(); \
fine_tuned or build_backbone(cfg)"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 "app.app:app"
