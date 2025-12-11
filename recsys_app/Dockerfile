# Use a slim python image to keep size reasonable
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (sometimes needed for pandas/numpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install python dependencies
# Note: This installs CPU+CUDA torch. 
# If you want a CPU-ONLY specific image (much smaller), use:
# RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]