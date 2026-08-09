# EnzySelect — educational prototype.
# Builds a self-contained image that runs with no API keys and no cloud account.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# curl is only needed for the container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so the layer caches across source edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The committed dataset is deterministic (fixed seed), so the image ships with
# data already present and needs no network at build or at run time. To
# regenerate it inside the container instead:
#     docker run --rm enzyselect python data/generate_data.py

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

# Cloud Run, Render, Railway, Heroku and Fly inject $PORT and expect the
# process to bind it. Default to 8501 so local `docker run -p 8501:8501`
# behaves exactly as before.
ENV PORT=8501
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail "http://localhost:${PORT:-8501}/_stcore/health" || exit 1

# `sh -c` so ${PORT} is expanded at runtime — the exec form (a JSON array)
# does not expand environment variables. `exec` then replaces the shell with
# Streamlit so it runs as PID 1 and receives SIGTERM directly, which keeps
# container shutdown prompt instead of waiting for the platform to SIGKILL.
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
