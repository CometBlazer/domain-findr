# Dockerfile
FROM python:3.11.9-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install system dependencies. curl is needed for HEALTHCHECK.
# gcc is removed as it's not needed for your current dependencies, making the image smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size. --use-pep517 is good practice.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Create a non-root user for better security
# This is a good practice you already had.
RUN useradd --system --create-home --uid 1001 appuser
USER appuser

# Expose the port your app will listen on.
# This is documentation; Cloud Run will map its PORT to this.
# It's good practice to have this match the default in your code (8000).
EXPOSE 8000

# Health check to ensure the API is responsive before marking the container as healthy.
# Note: The port here is hard-coded to 8000 because this runs *inside* the container,
# where we know the default port if $PORT isn't set.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# *** THE FIX IS HERE ***
# Run the application using a shell to interpret the $PORT environment variable.
# Google Cloud Run injects $PORT (usually 8080). If it's not set, it defaults to 8000.
CMD exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000}