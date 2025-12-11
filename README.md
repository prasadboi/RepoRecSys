# RepoRecSys
GitHub Repository Recommendation System

A production-ready recommendation system for GitHub repositories using Two-Tower Neural Collaborative Filtering, with FastAPI backend and React frontend.

## Project Structure

```
RepoRecSys/
├── code/
│   └── model/          # Model training code
├── data/               # Training and test datasets
├── frontend/           # React + TypeScript frontend
└── README.md
```

## Features

- 🎯 **Two-Tower Neural Collaborative Filtering**: Deep learning model for personalized recommendations
- 🚀 **FastAPI Backend**: High-performance REST API with async support
- ⚛️ **React Frontend**: Modern, responsive UI built with TypeScript
- ☁️ **Cloud-Ready**: Designed for deployment on Google Cloud Platform
- 🔄 **Incremental Training**: Support for periodic model updates
- 📊 **Real-time Inference**: Fast recommendation generation

## Quick Start

### Backend (FastAPI)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model (if needed):
```bash
python code/model/baseline_trainer.py
```

3. Start the API server:
```bash
python api.py
# Or: uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend (React)

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

- `POST /recommend` - Get repository recommendations for a user
  ```json
  {
    "user_id": 1,
    "top_k": 10
  }
  ```
- `GET /health` - Health check endpoint
- `GET /` - API information

## Deployment

### Frontend Deployment Options

The frontend can be deployed as:
- **Static Site**: Google Cloud Storage + CDN, Firebase Hosting, Netlify, Vercel
- **Containerized**: Cloud Run, Kubernetes
- **Integrated**: Served from FastAPI backend

See [frontend/DEPLOYMENT.md](frontend/DEPLOYMENT.md) for detailed deployment instructions.

### Backend Deployment

Deploy the FastAPI backend to:
- **Cloud Run**: Containerized deployment
- **Compute Engine**: VM-based deployment
- **Kubernetes**: For scalable deployments

See deployment documentation in the `u/loki-777/cloud-related-changes` branch for GCP deployment details.

## Architecture

```
┌─────────────┐         HTTP/REST          ┌─────────────┐
│   React     │ ────────────────────────> │   FastAPI   │
│  Frontend   │ <──────────────────────── │   Backend   │
│  (Static)   │         JSON Response     │  (Python)   │
└─────────────┘                            └─────────────┘
                                                  │
                                                  ▼
                                            ┌─────────────┐
                                            │   PyTorch   │
                                            │    Model    │
                                            └─────────────┘
```


