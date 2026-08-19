FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and benchmarks
COPY . .

# Expose API (8000) and Web UI (7860) ports
EXPOSE 8000
EXPOSE 7860

# Default command: Run FastAPI Server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
