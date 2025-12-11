FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY api.py .
COPY streamlit_app.py .

# Copy model files (these should be in the directory)
COPY best_twotower_model.pt .
COPY model_artifacts.pkl .
COPY item_metadata.pkl .

# Expose ports
EXPOSE 8000 8501

# Default command - run API
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

