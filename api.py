"""
FastAPI service for GitHub repository recommendations
"""
import os
import pickle
import torch
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from main import (
    Config, TwoTowerRecSys, NUMERIC_REPO_COLS, 
    normalize_numeric_features, ITEM_ID_COL, USER_ID_COL
)

app = FastAPI(title="GitHub Repository Recommendation API")

# Global variables for model and artifacts
model = None
artifacts = None
item_metadata = None
all_items_df = None
device = "cuda" if torch.cuda.is_available() else "cpu"

class RecommendationRequest(BaseModel):
    user_id: int
    top_k: int = 10

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[dict]  # List of {project_id, github_url, score, language, watchers}

def load_model_and_artifacts():
    """Load trained model and artifacts"""
    global model, artifacts, item_metadata, all_items_df
    
    print("Loading artifacts...")
    with open('model_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    try:
        with open('item_metadata.pkl', 'rb') as f:
            item_metadata = pickle.load(f)
    except FileNotFoundError:
        print("Warning: item_metadata.pkl not found, using defaults")
        item_metadata = {}
    
    print("Loading model...")
    cfg = Config()
    model = TwoTowerRecSys(
        num_users=artifacts['num_users'],
        num_items=artifacts['num_items'],
        num_langs=artifacts['num_langs'],
        num_numeric_feats=artifacts['num_numeric_feats'],
        cfg=cfg,
    ).to(device)
    
    model.load_state_dict(torch.load('best_twotower_model.pt', map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Load all items for candidate generation
    # Try to load from test data
    try:
        cfg = Config()
        test_bal = pd.read_csv(cfg.test_balanced_path)
        test_neg = pd.read_csv(cfg.test_negative_path)
        test_df = pd.concat([test_bal, test_neg], ignore_index=True)
        all_items_df = test_df.groupby(ITEM_ID_COL).first().reset_index()
        print(f"Loaded {len(all_items_df)} candidate items")
    except Exception as e:
        print(f"Warning: Could not load candidate items from test data: {e}")
        # Fallback: use items from artifacts
        if artifacts:
            all_items = list(artifacts['idx2item'].values())
            all_items_df = pd.DataFrame({ITEM_ID_COL: all_items})
            # Add default language_code
            all_items_df['language_code'] = 'unknown'
            # Add default numeric features (zeros, will be normalized)
            for col in NUMERIC_REPO_COLS:
                all_items_df[col] = 0.0
            print(f"Using {len(all_items_df)} items from artifacts")
        else:
            all_items_df = None

def get_github_url(project_id: int) -> str:
    """Convert project_id to GitHub URL"""
    # First try the mapping from artifacts
    if artifacts and project_id in artifacts.get('project_id_to_url', {}):
        url = artifacts['project_id_to_url'][project_id]
        if 'repo' not in url:  # If it's not a placeholder
            return url
    
    # Try to get from GitHub API (if project_id is a GitHub repo ID)
    try:
        # GitHub API endpoint for repository by ID
        response = requests.get(
            f"https://api.github.com/repositories/{project_id}",
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=5
        )
        if response.status_code == 200:
            repo_data = response.json()
            return repo_data.get('html_url', f"https://github.com/repo/{project_id}")
    except:
        pass
    
    # Fallback: construct URL (this might not work, but it's a placeholder)
    return f"https://github.com/repositories/{project_id}"

def recommend_for_user(user_id: int, top_k: int = 10) -> List[dict]:
    """Generate recommendations for a user"""
    if model is None or artifacts is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if user_id not in artifacts['user2idx']:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found in training data")
    
    # Get user index
    user_idx = artifacts['user2idx'][user_id]
    user_tensor = torch.tensor([user_idx], dtype=torch.long, device=device)
    
    # Get all candidate items
    if all_items_df is None:
        raise HTTPException(status_code=500, detail="No candidate items available")
    
    # Score all items
    item_ids = all_items_df[ITEM_ID_COL].values
    lang_codes = all_items_df['language_code'].values
    
    # Prepare tensors
    item_indices = torch.tensor(
        [artifacts['item2idx'][iid] for iid in item_ids],
        dtype=torch.long,
        device=device
    )
    lang_indices = torch.tensor(
        [artifacts['lang2idx'][lang] for lang in lang_codes],
        dtype=torch.long,
        device=device
    )
    
    # Normalize numeric features
    numerics = all_items_df[NUMERIC_REPO_COLS].copy()
    for col in NUMERIC_REPO_COLS:
        numerics[col] = (numerics[col].astype(float) - artifacts['means'][col]) / artifacts['stds'][col]
    numerics_tensor = torch.tensor(numerics.values, dtype=torch.float32, device=device)
    
    # Expand user tensor
    user_tensor_expanded = user_tensor.expand(len(item_ids))
    
    # Get predictions
    with torch.no_grad():
        logits, _, _ = model(user_tensor_expanded, item_indices, lang_indices, numerics_tensor)
        scores = torch.sigmoid(logits).detach().cpu().numpy()
    
    # Get top-K
    top_k_indices = np.argsort(-scores)[:top_k]
    
    # Build recommendations
    recommendations = []
    for idx in top_k_indices:
        project_id = item_ids[idx]
        score = float(scores[idx])
        github_url = get_github_url(project_id)
        
        # Get metadata
        meta = item_metadata.get(project_id, {})
        
        # Get language name from index
        lang_code = lang_codes[idx]
        lang_name = 'unknown'
        if 'idx2lang' in artifacts:
            lang_idx = int(lang_code) if isinstance(lang_code, (int, float)) else lang_code
            lang_name = artifacts['idx2lang'].get(lang_idx, 'unknown')
        
        # Use metadata language if available, otherwise use mapped name
        final_lang = meta.get('language_code', lang_name)
        if isinstance(final_lang, (int, float)):
            final_lang = str(final_lang)
        
        recommendations.append({
            'project_id': int(project_id),
            'github_url': github_url,
            'score': score,
            'language': final_lang,
            'watchers': int(meta.get('watchers', 0)),
        })
    
    return recommendations

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model_and_artifacts()

@app.get("/")
async def root():
    return {
        "message": "GitHub Repository Recommendation API",
        "endpoints": {
            "/recommend": "POST - Get recommendations for a user",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Get repository recommendations for a user"""
    try:
        recommendations = recommend_for_user(request.user_id, request.top_k)
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

