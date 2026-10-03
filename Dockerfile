# Antarctic Vessel Routing & Ice-Risk Forecasting System - API + dashboard image.
# CPU-only PyTorch keeps the image small enough for laptops and cloud VMs.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    ANTROUTE_CONFIG=/app/config/config.yaml \
    ANTROUTE_FIGURES=/app/docs/images

WORKDIR /app
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api,ml]" requests cdsapi netCDF4
# Full CMEMS client (heavier): pip install ".[ingest]"

COPY config ./config
COPY docs ./docs

RUN useradd --create-home --uid 1000 antroute && mkdir -p /app/data /app/reports /app/models \
    && chown -R antroute /app
USER antroute

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["antroute", "serve", "--host", "0.0.0.0", "--port", "8000", "--config", "/app/config/config.yaml"]
