FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (minimal since we removed complex deps)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .env .

# Expose port
EXPOSE 8000

# Run the application from src directory
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]