# Deployment Guide: GitHub Repository Recommendation System

## Overview
This guide walks you through deploying the GitHub repository recommendation API to Google Cloud Run.

## Prerequisites
1. Google Cloud account with billing enabled
2. Google Cloud SDK installed (`gcloud`)
3. Docker installed (for local testing)
4. Trained model files:
   - `best_twotower_model.pt`
   - `model_artifacts.pkl`
   - `item_metadata.pkl`

## Step 1: Prepare Model Files

After training, run:
```bash
python save_artifacts.py
```

This will create:
- `model_artifacts.pkl` - Contains mappings and scalers
- `item_metadata.pkl` - Contains item metadata

Make sure you have:
- `best_twotower_model.pt` - Trained model weights

## Step 2: Test Locally

### Test API locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Run API
python api.py
# Or: uvicorn api:app --host 0.0.0.0 --port 8000

# Test endpoint
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 10}'
```

### Test Streamlit frontend:
```bash
streamlit run streamlit_app.py
```

## Step 3: Build Docker Image

```bash
# Build image
docker build -t github-recommender .

# Test locally
docker run -p 8000:8000 github-recommender

# Test API
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 10}'
```

## Step 4: Deploy to Google Cloud Run

### Option A: Using gcloud CLI

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/github-recommender

# Deploy to Cloud Run
gcloud run deploy github-recommender \
  --image gcr.io/YOUR_PROJECT_ID/github-recommender \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300
```

### Option B: Using Cloud Build (automated)

```bash
# Submit build
gcloud builds submit --config cloudbuild.yaml
```

## Step 5: Update Streamlit App

Update the API URL in `streamlit_app.py`:
```python
API_URL = "https://YOUR_SERVICE_URL.run.app"
```

## Step 6: Deploy Streamlit (Optional)

### Option A: Streamlit Cloud (easiest)
1. Push code to GitHub
2. Go to streamlit.io
3. Connect repository
4. Deploy

### Option B: Cloud Run with Streamlit
Create a separate Dockerfile for Streamlit:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY streamlit_app.py .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Deploy:
```bash
gcloud run deploy github-recommender-ui \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Troubleshooting

### Model files not found
- Ensure `best_twotower_model.pt`, `model_artifacts.pkl`, and `item_metadata.pkl` are in the same directory as `api.py`

### Out of memory
- Increase Cloud Run memory: `--memory 4Gi`

### Slow inference
- Consider using GPU instances (Cloud Run doesn't support GPUs, use GKE or Compute Engine)
- Or optimize model (quantization, smaller batch sizes)

### GitHub API rate limits
- The `get_github_url` function uses GitHub API
- Consider caching URLs or using a mapping file instead

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /recommend` - Get recommendations
  ```json
  {
    "user_id": 1,
    "top_k": 10
  }
  ```

## Cost Estimation

- Cloud Run: ~$0.40 per million requests (first 2 million free)
- Storage: ~$0.02 per GB/month
- Estimated monthly cost: $5-20 (depending on usage)

## Next Steps

1. Add authentication if needed
2. Set up monitoring with Cloud Monitoring
3. Add caching for frequently requested users
4. Optimize model for faster inference

