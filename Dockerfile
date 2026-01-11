# Dockerfile
# Multi-stage build for optimization

# Stage 1: Base with system dependencies
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Optional: Install CUDA toolkit if GPU is available
ARG ENABLE_GPU=false
RUN if [ "$ENABLE_GPU" = "true" ]; then \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118; \
        pip install faiss-gpu; \
    else \
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
        pip install faiss-cpu; \
    fi

# Stage 3: Development
FROM dependencies as development

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/vector_store /app/data /app/logs

# Create non-root user for security
RUN useradd -m -u 1000 chatbot && \
    chown -R chatbot:chatbot /app
USER chatbot

# Expose port
EXPOSE 8000

# Default command (can be overridden by docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Production (optional)
FROM dependencies as production

COPY --from=development /app /app

# Remove development dependencies if needed
# RUN pip uninstall -y debugpy ipdb

USER chatbot

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
