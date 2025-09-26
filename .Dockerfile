# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if required for ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching layers)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (Flask app, templates, static, ML model, etc.)
COPY . .

# Hugging Face requires the app to run on port 7860
ENV PORT=7860

# Expose port
EXPOSE $PORT

# Run Flask (adjust if your entry point file is different)
# "app.py" should contain: app = Flask(__name__)
CMD ["python", "app.py"]
