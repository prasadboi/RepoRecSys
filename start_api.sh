#!/bin/bash
# Start the API server
# Run this to start the API

cd ~/github-recommender

# Check if files exist
if [ ! -f "api.py" ]; then
    echo "Error: api.py not found. Make sure you're in the right directory."
    exit 1
fi

if [ ! -f "best_twotower_model.pt" ]; then
    echo "Error: best_twotower_model.pt not found."
    exit 1
fi

if [ ! -f "model_artifacts.pkl" ]; then
    echo "Error: model_artifacts.pkl not found."
    exit 1
fi

# Start API
echo "Starting API server on port 8000..."
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000

