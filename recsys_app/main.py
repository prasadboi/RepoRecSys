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
SUBSCRIPTION_ID = "recsys-sub"
BUCKET_NAME = "recsys-data-bucket"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

publisher = pubsub_v1.PublisherClient()
subscriber = pubsub_v1.SubscriberClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

# --- GLOBAL STATE ---
ml_models = {}
artifacts = {}
model_lock = threading.Lock()
storage_client = storage.Client()

# --- HELPER: Global Top K ---
def compute_global_top_k(item_df: pd.DataFrame, k: int = 10):
    # Sort by 'events' (or watchers) to get global popularity
    top_items = item_df.sort_values("events", ascending=False).head(k)
    return top_items["project_id"].tolist()

# --- HELPER: Background Model re-loader ---
def check_for_model_update():
    while True:
        time.sleep(600)  # Sleep 10 mins
        load_latest_model_logic()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. SETUP GLOBAL DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts["device"] = device
    print(f"Device set to: {device}")

    # 2. SETUP PUBSUB INFRASTRUCTURE
    print("Setting up Pub/Sub...")
    try:
        try:
            publisher.create_topic(request={"name": topic_path})
        except Exception:
            pass 
        try:
            subscriber.create_subscription(request={"name": subscription_path, "topic": topic_path})
        except Exception:
            pass 
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
        
        # [FIX 1] Ensure IDs are strings and store the EXACT order used for inference
        item_df['project_id'] = item_df['project_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        ordered_item_ids = item_df['project_id'].tolist()
        
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
        for col in NUMERIC_COLS:
            m = scalers["means"].get(col, 0)
            s = scalers["stds"].get(col, 1)
            if col in item_df.columns:
                item_df[col] = (item_df[col] - m) / s
            else:
                item_df[col] = 0.0

        # [FIX 1] Use the ordered_item_ids for consistent embedding generation
        i_ids = torch.tensor([mappings["item2idx"].get(pid, 0) for pid in ordered_item_ids], device=device)
        l_ids = torch.tensor([mappings["lang2idx"].get(lc, 0) for lc in item_df["language_code"]], device=device)
        numerics = torch.tensor(item_df[NUMERIC_COLS].values, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            item_embs = model.item_tower(i_ids, l_ids, numerics)

        # Set Globals
        ml_models["model"] = model
        ml_models["item_embs"] = item_embs
        artifacts["user2idx"] = mappings["user2idx"]
        artifacts["idx2item"] = mappings["idx2item"]
        
        # [FIX 1] Store ordered IDs for lookup
        artifacts["ordered_item_ids"] = ordered_item_ids
        artifacts["global_top_k"] = global_top_k_ids
        
        print("Artifacts loaded successfully.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL STARTUP ERROR: {e}")
        artifacts["user2idx"] = {}
        artifacts["global_top_k"] = []
        artifacts["ordered_item_ids"] = []

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
    project_id: str  # Changed to str to match ID cleaning
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
        print(f"Unseen user {request.user_id}. triggering cold-start flow.")
        
        # 1. Publish to Pub/Sub
        message_json = json.dumps({
            "user_id": request.user_id,
            "user_name": request.user_name,
            "timestamp": time.time()
        })
        try:
            publisher.publish(topic_path, message_json.encode("utf-8"))
        except Exception as e:
            print(f"Failed to publish to Pub/Sub: {e}")

        # 2. Return Global Top K
        top_k_ids = artifacts["global_top_k"][:request.top_k]
        results = [
            Recommendation(project_id=str(pid), score=1.0, is_fallback=True) 
            for pid in top_k_ids
        ]
        return RecommendResponse(user_id=user_id_clean, recommendations=results)

    # --- EXISTING USER LOGIC ---
    with model_lock:
        model = ml_models["model"]
        item_embs = ml_models["item_embs"]
        
        u_idx = torch.tensor([user2idx[user_id_clean]], dtype=torch.long, device=device)
        with torch.no_grad():
            user_emb = model.user_tower(u_idx)
        
        scores = torch.matmul(user_emb, item_embs.T).squeeze(0)
        top_k_scores, top_k_indices = torch.topk(scores, k=request.top_k)

    # [FIX 1] Retrieve actual IDs using the ordered list from inference
    ordered_ids = artifacts["ordered_item_ids"]
    results = []
    for score, idx in zip(top_k_scores.cpu().tolist(), top_k_indices.cpu().tolist()):
        if idx < len(ordered_ids):
            results.append(Recommendation(
                project_id=ordered_ids[idx],
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
    print("--- Ingestion Task Started ---")
    # (Same ingestion logic as provided in source 1)
    # ... [Assuming standard ingestion logic from previous step] ...
    # For brevity, I am keeping the logic you already have. 
    # Just ensure to use the same logic as source 1.
    
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

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

    new_rows = []
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    
    defaults = {
        "mean_commits_language": 100.0, "max_commits_language": 5000.0,
        "mean_issues_language": 50.0, "mean_watchers_language": 200.0,
        "weight": 1.0, "cp": 0.0, "avg_cp": 0.0, "stddev": 0.0, "events": 1.0
    }

    for u in users_to_process:
        try:
            resp = requests.get(f"https://api.github.com/users/{u['user_name']}/starred", headers=headers)
            if resp.status_code != 200: continue
            
            repos = resp.json()
            for repo in repos:
                row = {
                    "id_user": u["user_id"],
                    "project_id": repo["id"],
                    "target": 1, 
                    "language_code": repo.get("language", "Unknown"),
                    "watchers": repo.get("stargazers_count", 0),
                    "issues": repo.get("open_issues_count", 0),
                    "pull_requests": repo.get("forks_count", 0),
                    "commits": 50,
                    "year": int(repo["created_at"][:4]) if repo.get("created_at") else 2024,
                }
                for col in NUMERIC_COLS:
                    if col not in row:
                        row[col] = defaults.get(col, 0.0)
                new_rows.append(row)
        except Exception as e:
            print(f"Error processing user {u['user_name']}: {e}")

    if new_rows:
        df = pd.DataFrame(new_rows)
        for c in NUMERIC_COLS:
            if c not in df.columns: df[c] = 0.0
            
        bucket = storage_client.bucket(BUCKET_NAME)
        timestamp = int(time.time())
        blob_name = f"training_data/batch_{timestamp}.csv"
        
        local_path = f"/tmp/batch_{timestamp}.csv"
        df.to_csv(local_path, index=False)
        bucket.blob(blob_name).upload_from_filename(local_path)
        print(f"Uploaded {len(df)} rows to {blob_name}")
    print("--- Ingestion Complete ---")


def train_task():
    """
    Downloads base data AND daily batches from GCS, calls pipeline, uploads results.
    """
    print("--- Training Task Started ---")
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # [FIX 2] Download Base Data from GCS
    # We download to /tmp/base_train.csv
    base_data_path = "/tmp/base_train.csv"
    print("Downloading base training data from GCS...")
    
    # Assumes the file is at 'archive/base_train.csv' or just 'base_train.csv'
    # Based on your previous setup, let's look in 'archive/' prefix if it exists, or root.
    # For now, let's assume you uploaded it to 'archive/base_train.csv'
    blob_base = bucket.blob("archive/base_train.csv") 
    
    if blob_base.exists():
        blob_base.download_to_filename(base_data_path)
        print(f"Downloaded base data to {base_data_path}")
    else:
        print("WARNING: 'archive/base_train.csv' not found in GCS. Training will rely ONLY on new batches.")
        # Ensure the path is deleted so train_utils doesn't read stale data
        if os.path.exists(base_data_path):
            os.remove(base_data_path)

    # 2. Download Daily Batches
    new_data_folder = "/tmp/new_data"
    if os.path.exists(new_data_folder):
        import shutil
        shutil.rmtree(new_data_folder)
    os.makedirs(new_data_folder, exist_ok=True)
    
    blobs = bucket.list_blobs(prefix="training_data/")
    count = 0
    for blob in blobs:
        if blob.name.endswith(".csv"):
            blob.download_to_filename(f"{new_data_folder}/{os.path.basename(blob.name)}")
            count += 1
    print(f"Downloaded {count} new batch files.")

    # 3. Run Training Pipeline
    # Pass the downloaded base_data_path
    success = train_model_pipeline(
        base_data_path=base_data_path, 
        new_data_folder=new_data_folder,
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
        blob_items = bucket.blob("latest_model/item_features.csv")
        
        if blob_model.exists() and blob_map.exists() and blob_items.exists():
            print("New model found. Downloading...")
            blob_model.download_to_filename("/tmp/temp_model.pt")
            blob_map.download_to_filename("/tmp/temp_mappings.pkl")
            blob_items.download_to_filename("/tmp/temp_items.csv")
            
            print("Hot-swapping model...")
            with model_lock:
                with open("/tmp/temp_mappings.pkl", "rb") as f:
                    mappings = pickle.load(f)
                
                # Load Items
                item_df = pd.read_csv("/tmp/temp_items.csv")
                
                # [FIX 1] Apply string cleaning and ID ordering logic
                item_df['project_id'] = item_df['project_id'].astype(str).str.replace(r'\.0$', '', regex=True)
                ordered_item_ids = item_df['project_id'].tolist()
                
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
                
                # Pre-compute new embeddings
                # We need scalers too. Ideally download them, but usually they don't shift drastically.
                # For correctness, we should have downloaded scalers.pkl too.
                # Assuming simple reload for now or that you add the scalers download above.
                # To prevent errors, we skip re-calc of embeddings if scalers missing, 
                # OR we download scalers in the block above (Added scalers download below).
                
                blob_scale = bucket.blob("latest_model/scalers.pkl")
                if blob_scale.exists():
                     blob_scale.download_to_filename("/tmp/temp_scalers.pkl")
                     with open("/tmp/temp_scalers.pkl", "rb") as f:
                        scalers = pickle.load(f)
                     
                     for col in NUMERIC_COLS:
                        m = scalers["means"].get(col, 0)
                        s = scalers["stds"].get(col, 1)
                        if col in item_df.columns:
                            item_df[col] = (item_df[col] - m) / s
                        else:
                            item_df[col] = 0.0

                     i_ids = torch.tensor([mappings["item2idx"].get(pid, 0) for pid in ordered_item_ids], device=device)
                     l_ids = torch.tensor([mappings["lang2idx"].get(lc, 0) for lc in item_df["language_code"]], device=device)
                     numerics = torch.tensor(item_df[NUMERIC_COLS].values, dtype=torch.float32, device=device)
                     
                     with torch.no_grad():
                        item_embs = new_model.item_tower(i_ids, l_ids, numerics)
                     
                     ml_models["item_embs"] = item_embs

                ml_models["model"] = new_model
                artifacts["user2idx"] = mappings["user2idx"]
                artifacts["idx2item"] = mappings["idx2item"]
                artifacts["ordered_item_ids"] = ordered_item_ids
                
            print("Model swapped successfully.")
            return True
        else:
            print("No new model found in GCS.")
            return False
    except Exception as e:
        print(f"Error in model reloader: {e}")
        return False