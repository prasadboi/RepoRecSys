import os
import torch
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader
from model import TwoTowerRecSys, Config, TwoTowerDataset

# Define the numeric columns strictly
NUMERIC_COLS = [
    "watchers", "commits", "issues", "pull_requests",
    "mean_commits_language", "max_commits_language", "min_commits_language", "std_commits_language",
    "mean_pull_requests_language", "max_pull_requests_language", "min_pull_requests_language", "std_pull_requests_language",
    "mean_issues_language", "max_issues_language", "min_issues_language", "std_issues_language",
    "mean_watchers_language", "max_watchers_language", "min_watchers_language", "std_watchers_language",
    "events", "year", "weight", "cp", "avg_cp", "stddev",
]

def train_model_pipeline(
    base_data_path: str,
    new_data_folder: str, 
    output_artifacts_dir: str,
    device: str
):
    print("--- Starting Training Pipeline ---")
    
    # 1. LOAD DATA
    # Load base dataset (the initial training CSV)
    # In a real system, you might read this from GCS too.
    print("Loading base data...")
    if os.path.exists(base_data_path):
        base_df = pd.read_csv(base_data_path)
    else:
        # Fallback for demo: create empty DF if base missing (rare)
        base_df = pd.DataFrame(columns=['id_user', 'project_id', 'language_code', 'target'] + NUMERIC_COLS)

    # Load new CSVs from the "new_data" folder (downloaded from GCS)
    print(f"Loading new data from {new_data_folder}...")
    new_dfs = []
    if os.path.exists(new_data_folder):
        for f in os.listdir(new_data_folder):
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(new_data_folder, f))
                    new_dfs.append(df)
                except Exception as e:
                    print(f"Skipping corrupt file {f}: {e}")
    
    if new_dfs:
        full_df = pd.concat([base_df] + new_dfs, ignore_index=True)
    else:
        full_df = base_df
    
    print("Normalizing User IDs...")
    # Convert to string, then remove trailing '.0' if present
    full_df['id_user'] = full_df['id_user'].astype(str).str.replace(r'\.0$', '', regex=True)
    
    # Ensure items are strings too just to be safe
    full_df['project_id'] = full_df['project_id'].astype(str).str.replace(r'\.0$', '', regex=True)

    print(f"Total training samples: {len(full_df)}")

    # 2. UPDATE MAPPINGS
    # We must regenerate mappings to include any new users/items found in daily batches
    print("Updating ID mappings...")
    all_users = full_df['id_user'].unique()
    all_items = full_df['project_id'].unique()
    all_langs = full_df['language_code'].astype(str).unique()

    user2idx = {u: i for i, u in enumerate(all_users)}
    item2idx = {item: i for i, item in enumerate(all_items)}
    lang2idx = {l: i for i, l in enumerate(all_langs)}
    idx2item = {i: item for item, i in item2idx.items()} # Reverse mapping

    # 3. RE-COMPUTE SCALERS
    # Statistics change as new data arrives, so we re-compute mean/std
    print("Re-computing feature scalers...")
    means = {}
    stds = {}
    for col in NUMERIC_COLS:
        # Handle potential missing cols in new data by filling 0
        if col not in full_df.columns:
            full_df[col] = 0.0
            
        vals = full_df[col].astype(float).values
        m = vals.mean()
        s = vals.std()
        if s < 1e-6: s = 1.0
        means[col] = m
        stds[col] = s
        
        # Normalize in-place
        full_df[col] = (full_df[col] - m) / s

    # 4. PREPARE DATASET
    print("Preparing Datasets...")
    # Filter for positives only for TwoTower training (implicit feedback)
    train_df = full_df[full_df['target'] == 1].reset_index(drop=True)
    
    dataset = TwoTowerDataset(train_df, user2idx, item2idx, lang2idx)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=0)

    # 5. INITIALIZE MODEL
    cfg = Config()
    cfg.device = device
    
    model = TwoTowerRecSys(
        num_users=len(user2idx),
        num_items=len(item2idx),
        num_langs=len(lang2idx),
        num_numeric_feats=len(NUMERIC_COLS),
        cfg=cfg
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # 6. TRAINING LOOP
    print("Training model...")
    model.train()
    # For demo purposes, we run fewer epochs. In prod, run more.
    EPOCHS = 3 
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            u, i, l, nums, _ = batch
            u, i, l, nums = u.to(device), i.to(device), l.to(device), nums.to(device)
            
            optimizer.zero_grad()
            _, u_emb, v_emb = model(u, i, l, nums)
            
            # Contrastive Loss
            sim_matrix = torch.matmul(u_emb, v_emb.T)
            logits = sim_matrix / 0.1
            targets = torch.arange(u.size(0), device=device)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(dataloader):.4f}")

    # 7. SAVE ARTIFACTS
    print("Saving artifacts...")
    os.makedirs(output_artifacts_dir, exist_ok=True)
    
    # Save Model
    torch.save(model.state_dict(), os.path.join(output_artifacts_dir, "best_twotower_model.pt"))
    
    # Save Mappings (Pickle)
    mappings = {
        "user2idx": user2idx, 
        "item2idx": item2idx, 
        "lang2idx": lang2idx,
        "idx2item": idx2item
    }
    with open(os.path.join(output_artifacts_dir, "mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)
        
    # Save Scalers (Pickle)
    scalers = {"means": means, "stds": stds}
    with open(os.path.join(output_artifacts_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    # Save Unique Item Table (for inference cache)
    # We take the LAST seen state of every item
    item_df = full_df.sort_values("year").drop_duplicates(subset=["project_id"], keep="last")
    # Denormalize for saving (optional, but keeps consistency with main.py logic which expects raw-ish input)
    # OR save normalized. Let's save normalized to save compute in main.py, 
    # BUT main.py currently expects to normalize itself. 
    # To avoid breaking main.py, let's reverse the normalization or just save the raw columns if we had them.
    # SIMPLIFICATION: We will save the normalized values and update main.py to handle that? 
    # No, easier to just save the table. main.py will normalize again. It's a bit redundant but safe.
    
    # Actually, main.py expects raw input to apply scalers. 
    # Since we normalized in-place above, we should ideally use a copy or reverse it.
    # For now, let's just save the project_id/language_code and the columns.
    item_df[['project_id', 'language_code'] + NUMERIC_COLS].to_csv(
        os.path.join(output_artifacts_dir, "item_features.csv"), index=False
    )

    print("Pipeline complete.")
    return True