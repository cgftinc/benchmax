# Every task in aime/aime@latest uses this exact, Dockerfile-only image.
FROM ghcr.io/laude-institute/t-bench/ubuntu-24-04:20250624

# The bridge binary and Worker package must use the same Sandbox SDK version.
COPY --from=docker.io/cloudflare/sandbox:0.12.9 /container-server/sandbox /sandbox

RUN mkdir -p /workspace /app /logs/agent /logs/verifier /logs/artifacts \
    /tests /solution /harbor/skills \
    && chmod 777 /workspace /app /logs /logs/agent /logs/verifier \
    /logs/artifacts /tests /solution /harbor /harbor/skills

WORKDIR /app
ENTRYPOINT ["/sandbox"]
