import os
import json
import pickle
import threading
import time
import requests
import datetime
import torch
import pandas as pd
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from google.cloud import pubsub_v1
from google.cloud import storage

from model import TwoTowerRecSys, Config
from train_utils import train_model_pipeline, NUMERIC_COLS

# --- CONFIG ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "my-project-id")
TOPIC_ID = "new-user-events"
SUBSCRIPTION_ID = "recsys-sub" # Define explicitly
BUCKET_NAME = "recsys-data-bucket"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient() # Initialize here for global use
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

# --- GLOBAL STATE ---
ml_models = {}
artifacts = {}
model_lock = threading.Lock()


storage_client = storage.Client()

# --- HELPER: Global Top K ---
def compute_global_top_k(item_df: pd.DataFrame, k: int = 10):
    """
    Simple heuristic: Sort by 'watchers' or 'events' to get global popularity.
    """
    # Assuming 'events' correlates with popularity
    top_items = item_df.sort_values("events", ascending=False).head(k)
    return top_items["project_id"].tolist()

# --- HELPER: Background Model re-loader ---
def check_for_model_update():
    while True:
        time.sleep(600)  # Sleep 10 mins
        load_latest_model_logic()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. SETUP GLOBAL DEVICE IMMEDIATELY (Prevent KeyError)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts["device"] = device
    print(f"Device set to: {device}")

    # 2. SETUP PUBSUB INFRASTRUCTURE
    print("Setting up Pub/Sub...")
    try:
        try:
            publisher.create_topic(request={"name": topic_path})
        except Exception:
            pass # Topic exists
        try:
            subscriber.create_subscription(request={"name": subscription_path, "topic": topic_path})
        except Exception:
            pass # Subscription exists
    except Exception as e:
        print(f"Warning: Infrastructure setup failed: {e}")

    # 3. LOAD ARTIFACTS
    print("Loading artifacts...")
    try:
        base_path = "artifacts"

        # Load Mappings
        with open(f"{base_path}/mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
        
        # Load Scalers
        with open(f"{base_path}/scalers.pkl", "rb") as f:
            scalers = pickle.load(f)
        
        # Load Item Features
        item_df = pd.read_csv(f"{base_path}/item_features.csv")
        global_top_k_ids = compute_global_top_k(item_df)
        
        # Load Model
        cfg = Config()
        cfg.device = device
        model = TwoTowerRecSys(
            len(mappings["user2idx"]), 
            len(mappings["item2idx"]), 
            len(mappings["lang2idx"]), 
            len(NUMERIC_COLS), 
            cfg
        )
        
        model_path = f"{base_path}/best_twotower_model.pt"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
        else:
            print(f"Warning: Model file not found at {model_path}")

        # Pre-compute Embeddings
        # Normalize features
        for col in NUMERIC_COLS:
            m = scalers["means"].get(col, 0)
            s = scalers["stds"].get(col, 1)
            if col in item_df.columns:
                item_df[col] = (item_df[col] - m) / s
            else:
                item_df[col] = 0.0

        i_ids = torch.tensor([mappings["item2idx"].get(str(pid), 0) for pid in item_df["project_id"]], device=device)
        l_ids = torch.tensor([mappings["lang2idx"].get(lc, 0) for lc in item_df["language_code"]], device=device)
        numerics = torch.tensor(item_df[NUMERIC_COLS].values, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            item_embs = model.item_tower(i_ids, l_ids, numerics)

        # Set Globals
        ml_models["model"] = model
        ml_models["item_embs"] = item_embs
        artifacts["user2idx"] = mappings["user2idx"]
        artifacts["idx2item"] = mappings["idx2item"]
        artifacts["global_top_k"] = global_top_k_ids
        
        print("Artifacts loaded successfully.")

    except Exception as e:
        # PRINT THE REAL ERROR
        import traceback
        traceback.print_exc()
        print(f"CRITICAL STARTUP ERROR: {e}")
        # Initialize empty defaults to prevent crashing
        artifacts["user2idx"] = {}
        artifacts["global_top_k"] = []

    # Start reloader
    threading.Thread(target=check_for_model_update, daemon=True).start()
    
    print("System Ready.")
    yield
    ml_models.clear()

app = FastAPI(title="GCP RecSys", lifespan=lifespan)

class RecommendRequest(BaseModel):
    user_id: str
    user_name: str
    top_k: int = 10

class Recommendation(BaseModel):
    project_id: int
    score: float
    is_fallback: bool = False

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[Recommendation]

# --- ENDPOINTS ---

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    user2idx = artifacts["user2idx"]
    device = artifacts["device"]
    user_id_clean = str(request.user_id).replace(".0", "")
    
    # CHECK: Is user known?
    if user_id_clean not in user2idx:
        # --- UNSEEN USER LOGIC ---
        print(f"Unseen user {request.user_id}. triggering cold-start flow.")
        
        # 1. Publish to Pub/Sub
        message_json = json.dumps({
            "user_id": request.user_id,
            "user_name": request.user_name,
            "timestamp": time.time()
        })
        try:
            future = publisher.publish(topic_path, message_json.encode("utf-8"))
            # future.result() # Don't wait for result to reduce latency
        except Exception as e:
            print(f"Failed to publish to Pub/Sub: {e}")

        # 2. Return Global Top K
        top_k_ids = artifacts["global_top_k"][:request.top_k]
        results = [
            Recommendation(project_id=pid, score=1.0, is_fallback=True) 
            for pid in top_k_ids
        ]
        return RecommendResponse(user_id=user_id_clean, recommendations=results)

    # --- EXISTING USER LOGIC ---
    # Thread-safe model access
    with model_lock:
        model = ml_models["model"]
        item_embs = ml_models["item_embs"]
        
        u_idx = torch.tensor([user2idx[user_id_clean]], dtype=torch.long, device=device)
        with torch.no_grad():
            user_emb = model.user_tower(u_idx)
        
        scores = torch.matmul(user_emb, item_embs.T).squeeze(0)
        top_k_scores, top_k_indices = torch.topk(scores, k=request.top_k)

    idx2item = artifacts["idx2item"]
    results = []
    for score, idx in zip(top_k_scores.cpu().tolist(), top_k_indices.cpu().tolist()):
        results.append(Recommendation(
            project_id=idx2item.get(idx, -1),
            score=score
        ))

    return RecommendResponse(user_id=request.user_id, recommendations=results)


@app.post("/system/ingest")
def trigger_ingest(background_tasks: BackgroundTasks):
    background_tasks.add_task(ingest_task)
    return {"status": "Ingestion started"}

@app.post("/system/train")
def trigger_train(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_task)
    return {"status": "Training started"}


@app.post("/system/reload")
def manual_reload():
    """Forces the server to check GCS and reload the model immediately."""
    success = load_latest_model_logic()
    if success:
        return {"status": "Model reloaded successfully"}
    else:
        return {"status": "No new model found or update failed"}


def ingest_task():
    """
    Pulls user IDs from Pub/Sub, fetches their starred repos from GitHub,
    constructs the training CSV row, and uploads to GCS.
    """
    print("--- Ingestion Task Started ---")
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, "recsys-sub")

    # Pull Messages
    users_to_process = []
    try:
        response = subscriber.pull(request={"subscription": subscription_path, "max_messages": 50})
        ack_ids = []
        for msg in response.received_messages:
            data = json.loads(msg.message.data.decode("utf-8"))
            users_to_process.append(data)
            ack_ids.append(msg.ack_id)
        if ack_ids:
            subscriber.acknowledge(request={"subscription": subscription_path, "ack_ids": ack_ids})
    except Exception as e:
        print(f"PubSub Pull Error: {e}")
        return

    if not users_to_process:
        print("No new users.")
        return

    # Fetch Data
    new_rows = []
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    
    # We define defaults for the complex aggregate columns we can't calc on the fly
    # In a real app, you might fetch these from a "Language Stats" database.
    defaults = {
        "mean_commits_language": 100.0, "max_commits_language": 5000.0,
        "mean_issues_language": 50.0, "mean_watchers_language": 200.0,
        "weight": 1.0, "cp": 0.0, "avg_cp": 0.0, "stddev": 0.0, "events": 1.0
    }

    for u in users_to_process:
        try:
            # 1. Get Starred Repos
            resp = requests.get(f"https://api.github.com/users/{u['user_name']}/starred", headers=headers)
            if resp.status_code != 200: continue
            
            repos = resp.json()
            for repo in repos:
                # 2. Extract Features
                row = {
                    "id_user": u["user_id"],
                    "project_id": repo["id"],
                    "target": 1, # Implicit positive
                    "language_code": repo.get("language", "Unknown"),
                    
                    # Direct API mappings
                    "watchers": repo.get("stargazers_count", 0),
                    "issues": repo.get("open_issues_count", 0),
                    "pull_requests": repo.get("forks_count", 0), # Proxy
                    "commits": 50, # GitHub API doesn't give commits count in summary. Mocking.
                    
                    # Year
                    "year": int(repo["created_at"][:4]) if repo.get("created_at") else 2024,
                }
                
                # Fill complex columns with defaults
                for col in NUMERIC_COLS:
                    if col not in row:
                        row[col] = defaults.get(col, 0.0)
                        
                new_rows.append(row)
        except Exception as e:
            print(f"Error processing user {u['user_name']}: {e}")

    # Upload to GCS
    if new_rows:
        df = pd.DataFrame(new_rows)
        # Ensure we have all columns
        for c in NUMERIC_COLS:
            if c not in df.columns: df[c] = 0.0
            
        bucket = storage_client.bucket(BUCKET_NAME)
        timestamp = int(time.time())
        blob_name = f"training_data/batch_{timestamp}.csv"
        
        # Save locally then upload
        local_path = f"/tmp/batch_{timestamp}.csv"
        df.to_csv(local_path, index=False)
        bucket.blob(blob_name).upload_from_filename(local_path)
        print(f"Uploaded {len(df)} rows to {blob_name}")
    print("--- Ingestion Complete ---")


def train_task():
    """
    Downloads all data from GCS, calls train_model_pipeline, uploads results.
    """
    print("--- Training Task Started ---")
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # 1. Download Base Data (if stored in GCS, otherwise we rely on local image artifact)
    # For this setup, we assume base data is available at 'artifacts/item_features.csv' 
    # or similar, but ideally we download a 'base_train.csv'
    
    # 2. Download Daily Batches
    os.makedirs("/tmp/new_data", exist_ok=True)
    blobs = bucket.list_blobs(prefix="training_data/")
    for blob in blobs:
        if blob.name.endswith(".csv"):
            blob.download_to_filename(f"/tmp/new_data/{os.path.basename(blob.name)}")
            
    # 3. Run Training Pipeline
    # Using 'artifacts/train_balanced.csv' if you included it in image, 
    # otherwise use the 'item_features' as a weak proxy or download real base.
    success = train_model_pipeline(
        base_data_path="archive/train_balanced.csv", # Assumes this is in image
        new_data_folder="/tmp/new_data",
        output_artifacts_dir="/tmp/output_artifacts",
        device=artifacts["device"]
    )
    
    if success:
        # 4. Upload New Artifacts to GCS
        print("Uploading new models to GCS...")
        prefix = "latest_model"
        for f_name in ["best_twotower_model.pt", "mappings.pkl", "scalers.pkl", "item_features.csv"]:
            local_f = os.path.join("/tmp/output_artifacts", f_name)
            if os.path.exists(local_f):
                bucket.blob(f"{prefix}/{f_name}").upload_from_filename(local_f)
        print("Training and Upload Complete.")


def load_latest_model_logic():
    print("Checking for new model in GCS...")
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_model = bucket.blob("latest_model/best_twotower_model.pt")
        blob_map = bucket.blob("latest_model/mappings.pkl")
        
        if blob_model.exists() and blob_map.exists():
            print("New model found. Downloading...")
            blob_model.download_to_filename("/tmp/temp_model.pt")
            blob_map.download_to_filename("/tmp/temp_mappings.pkl")
            
            print("Hot-swapping model...")
            with model_lock:
                with open("/tmp/temp_mappings.pkl", "rb") as f:
                    mappings = pickle.load(f)
                
                cfg = Config()
                device = artifacts["device"]
                
                new_model = TwoTowerRecSys(
                    len(mappings["user2idx"]), 
                    len(mappings["item2idx"]), 
                    len(mappings["lang2idx"]), 
                    len(NUMERIC_COLS), 
                    cfg
                )
                state_dict = torch.load("/tmp/temp_model.pt", map_location=device)
                new_model.load_state_dict(state_dict)
                new_model.to(device)
                new_model.eval()
                
                ml_models["model"] = new_model
                artifacts["user2idx"] = mappings["user2idx"]
                artifacts["idx2item"] = mappings["idx2item"]
                
            print("Model swapped successfully.")
            return True
        else:
            print("No new model found in GCS.")
            return False
    except Exception as e:
        print(f"Error in model reloader: {e}")
        return False