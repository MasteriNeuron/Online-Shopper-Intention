# Use a lightweight Python base image
FROM python:3.10-slim

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install system dependencies required for Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (Flask app, templates, static, ML model, etc.)
COPY . .

# Create directories for runtime files and set full permissions
RUN mkdir -p logs temp output datasets/processed models static/plots \
    && chown -R appuser:appuser /app/logs /app/temp /app/output /app/datasets /app/models /app/static/plots \
    && chmod -R 777 /app/logs /app/temp /app/output /app/datasets /app/models /app/static/plots

# Set environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port 7860
EXPOSE 7860

# Switch to non-root user
USER appuser

# Run the Flask app
CMD ["python", "app.py"]
