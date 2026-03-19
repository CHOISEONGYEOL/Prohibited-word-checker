FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (saves ~1.5GB vs full CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model during build (avoids runtime download)
ENV SENTENCE_TRANSFORMERS_HOME=/app/.model_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# Clean up build dependencies
RUN apt-get purge -y gcc g++ && apt-get autoremove -y && \
    rm -rf /root/.cache/pip

# Copy application code
COPY . .

# Remove unnecessary files
RUN rm -rf .git* __pycache__ .env

EXPOSE 8000

ENV SENTENCE_TRANSFORMERS_HOME=/app/.model_cache

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
