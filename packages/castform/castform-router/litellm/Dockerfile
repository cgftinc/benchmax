FROM ghcr.io/berriai/litellm:v1.94.0

WORKDIR /opt/castform-router
COPY pyproject.toml README.md ./
COPY castform_router ./castform_router
RUN pip install --no-cache-dir .

COPY litellm/config.yaml /app/castform-router.yaml
CMD ["--config", "/app/castform-router.yaml", "--port", "4000"]
