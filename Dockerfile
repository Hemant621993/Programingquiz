FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p webcam_captures temp

# Expose port for the Flask app
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--reuse-port", "--reload", "main:app"]