# -*- coding: utf-8 -*-
"""
GitHub Repository Recommendation System - Two Tower Model
Training script for the recommendation system
"""

# github_recsys_twotower_with_val_split.py

import os
import math
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Optional: for AUC metric
try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


########################################
# 1. CONFIG AND FEATURE DEFINITIONS
########################################

class Config:
    # File paths: adapt these to your actual locations
    # Update these paths to point to your data files
    data_dir = "./data"  # Change this to your data directory
    train_balanced_path = f"{data_dir}/train_balanced.csv"
    train_negative_path = f"{data_dir}/train_negative.csv"
    test_balanced_path = f"{data_dir}/test_balanced.csv"
    test_negative_path = f"{data_dir}/test_negative.csv"

    # Model hyperparameters
    user_id_emb_dim = 64
    item_id_emb_dim = 64
    lang_emb_dim = 16
    hidden_dim = 128
    embedding_dim = 64  # final tower output dimension

    batch_size = 2048
    num_epochs = 5
    lr = 1e-3
    weight_decay = 1e-5

    # Validation split
    train_user_fraction = 0.8  # 80% users for train, 20% for validation

    # CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Random seeds for reproducibility
    seed = 42


# Columns in test_balanced.csv
# ['id_user', 'project_id', 'watchers', 'commits', 'issues', 'pull_requests',
#  'target', 'mean_commits_language', 'max_commits_language',
#  'min_commits_language', 'std_commits_language',
#  'mean_pull_requests_language', 'max_pull_requests_language',
#  'min_pull_requests_language', 'std_pull_requests_language',
#  'mean_issues_language', 'max_issues_language', 'min_issues_language',
#  'std_issues_language', 'mean_watchers_language', 'max_watchers_language',
#  'min_watchers_language', 'std_watchers_language', 'events', 'year',
#  'language_code', 'weight', 'cp', 'avg_cp', 'stddev']

NUMERIC_REPO_COLS = [
    "watchers", "commits", "issues", "pull_requests",
    "mean_commits_language", "max_commits_language", "min_commits_language",
    "std_commits_language",
    "mean_pull_requests_language", "max_pull_requests_language",
    "min_pull_requests_language", "std_pull_requests_language",
    "mean_issues_language", "max_issues_language", "min_issues_language",
    "std_issues_language",
    "mean_watchers_language", "max_watchers_language",
    "min_watchers_language", "std_watchers_language",
    "events", "year",
    "weight", "cp", "avg_cp", "stddev",
]

CATEGORICAL_REPO_COLS = [
    "language_code",
]

USER_ID_COL = "id_user"
ITEM_ID_COL = "project_id"
TARGET_COL = "target"

########################################
# 2. DATA LOADING AND PREPROCESSING
########################################

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_raw_data(cfg: Config):
    train_bal = pd.read_csv(cfg.train_balanced_path)
    train_neg = pd.read_csv(cfg.train_negative_path)
    test_bal = pd.read_csv(cfg.test_balanced_path)
    test_neg = pd.read_csv(cfg.test_negative_path)
    return train_bal, train_neg, test_bal, test_neg


def build_id_mappings(train_pool: pd.DataFrame, test_df: pd.DataFrame):
    # Unique users and items across train_pool + test
    all_users = pd.concat([train_pool[USER_ID_COL], test_df[USER_ID_COL]]).unique()
    all_items = pd.concat([train_pool[ITEM_ID_COL], test_df[ITEM_ID_COL]]).unique()
    all_langs = pd.concat([train_pool["language_code"], test_df["language_code"]]).unique()

    user2idx = {uid: i for i, uid in enumerate(all_users)}
    item2idx = {iid: i for i, iid in enumerate(all_items)}
    lang2idx = {lid: i for i, lid in enumerate(all_langs)}

    idx2user = {i: uid for uid, i in user2idx.items()}
    idx2item = {i: iid for iid, i in item2idx.items()}
    idx2lang = {i: lid for lid, i in lang2idx.items()}

    return user2idx, item2idx, lang2idx, idx2user, idx2item, idx2lang


def compute_numeric_scalers(train_pool: pd.DataFrame):
    """
    Compute mean and std for each numeric repo feature on training pool data.
    Returns dicts: col -> (mean, std).
    """
    means = {}
    stds = {}
    for col in NUMERIC_REPO_COLS:
        col_values = train_pool[col].astype(float).values
        mean = col_values.mean()
        std = col_values.std()
        if std < 1e-6:
            std = 1.0
        means[col] = mean
        stds[col] = std
    return means, stds


def normalize_numeric_features(df: pd.DataFrame, means: Dict[str, float], stds: Dict[str, float]):
    df = df.copy()
    for col in NUMERIC_REPO_COLS:
        df[col] = (df[col].astype(float) - means[col]) / stds[col]
    return df


def split_train_val_by_user(
    train_pool: pd.DataFrame,
    train_user_fraction: float,
    seed: int,
):
    """
    Split the training pool into train_df and val_df based on users.
    """
    all_users = train_pool[USER_ID_COL].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(all_users)

    n_train_users = int(len(all_users) * train_user_fraction)
    train_users = set(all_users[:n_train_users])
    val_users = set(all_users[n_train_users:])

    train_df = train_pool[train_pool[USER_ID_COL].isin(train_users)].reset_index(drop=True)
    val_df = train_pool[train_pool[USER_ID_COL].isin(val_users)].reset_index(drop=True)

    return train_df, val_df

########################################
# 3. DATASET AND DATALOADER
########################################

class TwoTowerDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        user2idx: Dict[int, int],
        item2idx: Dict[int, int],
        lang2idx: Dict[int, int],
    ):
        self.df = df.reset_index(drop=True)
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.lang2idx = lang2idx

        # Pre-extract numpy arrays so __getitem__ is fast
        self.user_ids = self.df[USER_ID_COL].values
        self.item_ids = self.df[ITEM_ID_COL].values
        self.lang_codes = self.df["language_code"].values
        self.labels = self.df[TARGET_COL].astype(float).values

        self.numeric_matrix = self.df[NUMERIC_REPO_COLS].astype(float).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        uid = self.user_ids[idx]
        iid = self.item_ids[idx]
        lang = self.lang_codes[idx]
        label = self.labels[idx]
        numerics = self.numeric_matrix[idx]

        u_idx = self.user2idx[uid]
        i_idx = self.item2idx[iid]
        l_idx = self.lang2idx[lang]

        # Convert to tensors
        u_idx = torch.tensor(u_idx, dtype=torch.long)
        i_idx = torch.tensor(i_idx, dtype=torch.long)
        l_idx = torch.tensor(l_idx, dtype=torch.long)
        numerics = torch.tensor(numerics, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return u_idx, i_idx, l_idx, numerics, label


def make_dataloaders(
    cfg: Config,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    user2idx: Dict[int, int],
    item2idx: Dict[int, int],
    lang2idx: Dict[int, int],
):
    train_dataset = TwoTowerDataset(train_df, user2idx, item2idx, lang2idx)
    val_dataset = TwoTowerDataset(val_df, user2idx, item2idx, lang2idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader

########################################
# 4. MODEL DEFINITIONS
########################################

class UserTower(nn.Module):
    def __init__(self, num_users: int, cfg: Config):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, cfg.user_id_emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(cfg.user_id_emb_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        user_ids: [B]
        returns: [B, embedding_dim]
        """
        x = self.user_emb(user_ids)  # [B, user_id_emb_dim]
        x = self.mlp(x)              # [B, embedding_dim]
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x


class ItemTower(nn.Module):
    def __init__(self, num_items: int, num_langs: int, num_numeric_feats: int, cfg: Config):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, cfg.item_id_emb_dim)
        self.lang_emb = nn.Embedding(num_langs, cfg.lang_emb_dim)

        input_dim = cfg.item_id_emb_dim + cfg.lang_emb_dim + num_numeric_feats

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )

    def forward(
        self,
        item_ids: torch.Tensor,     # [B]
        lang_ids: torch.Tensor,     # [B]
        numeric_feats: torch.Tensor # [B, num_numeric_feats]
    ) -> torch.Tensor:
        """
        returns: [B, embedding_dim]
        """
        item_e = self.item_emb(item_ids)    # [B, item_id_emb_dim]
        lang_e = self.lang_emb(lang_ids)    # [B, lang_emb_dim]

        x = torch.cat([item_e, lang_e, numeric_feats], dim=-1)  # [B, input_dim]
        x = self.mlp(x)                                         # [B, embedding_dim]
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x


class TwoTowerRecSys(nn.Module):
    def __init__(self, num_users: int, num_items: int, num_langs: int, num_numeric_feats: int, cfg: Config):
        super().__init__()
        self.user_tower = UserTower(num_users, cfg)
        self.item_tower = ItemTower(num_items, num_langs, num_numeric_feats, cfg)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        numeric_feats: torch.Tensor,
    ):
        """
        user_ids: [B]
        item_ids: [B]
        lang_ids: [B]
        numeric_feats: [B, F]

        returns:
            logits: [B] (dot products)
            user_embs: [B, D]
            item_embs: [B, D]
        """
        u = self.user_tower(user_ids)                          # [B, D]
        v = self.item_tower(item_ids, lang_ids, numeric_feats) # [B, D]

        logits = torch.sum(u * v, dim=-1)  # [B]
        return logits, u, v

########################################
# 5. TRAINING AND EVALUATION
########################################

def train_one_epoch(
    model: TwoTowerRecSys,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
):
    model.train()
    total_loss = 0.0
    total_examples = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch in train_loader:
        user_ids, item_ids, lang_ids, numerics, labels = batch
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        lang_ids = lang_ids.to(device)
        numerics = numerics.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, u, v = model(user_ids, item_ids, lang_ids, numerics)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    avg_loss = total_loss / max(total_examples, 1)
    return avg_loss


def evaluate_pointwise(
    model: TwoTowerRecSys,
    data_loader: DataLoader,
    device: str,
):
    """
    Simple pointwise evaluation:
    - average loss
    - ROC-AUC (if sklearn available)
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_examples = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            user_ids, item_ids, lang_ids, numerics, labels = batch
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            lang_ids = lang_ids.to(device)
            numerics = numerics.to(device)
            labels = labels.to(device)

            logits, u, v = model(user_ids, item_ids, lang_ids, numerics)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

    avg_loss = total_loss / max(total_examples, 1)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    if SKLEARN_AVAILABLE:
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")

    return avg_loss, auc


def evaluate_hit_at_k(
    model: TwoTowerRecSys,
    data_df: pd.DataFrame,
    user2idx: Dict[int, int],
    item2idx: Dict[int, int],
    lang2idx: Dict[int, int],
    means: Dict[str, float],
    stds: Dict[str, float],
    device: str,
    k: int = 10,
):
    """
    Simple Hit@K over a dataframe:
    For each user in data_df, we:
      - take all (repo, label) pairs for that user in data_df,
      - score each repo with the model,
      - rank repos by score,
      - check if any positive label appears in top-K.
    Returns average Hit@K across users.
    """
    model.eval()

    grouped = data_df.groupby(USER_ID_COL)

    hits = []
    with torch.no_grad():
        for uid, group in grouped:
            labels = group[TARGET_COL].values
            if labels.max() == 0:
                # no positive in this group; skip
                continue

            u_idx = user2idx[uid]
            u_tensor = torch.tensor([u_idx], dtype=torch.long, device=device)

            item_ids_raw = group[ITEM_ID_COL].values
            lang_codes_raw = group["language_code"].values
            item_indices = torch.tensor(
                [item2idx[i] for i in item_ids_raw],
                dtype=torch.long,
                device=device,
            )
            lang_indices = torch.tensor(
                [lang2idx[l] for l in lang_codes_raw],
                dtype=torch.long,
                device=device,
            )

            numerics = group[NUMERIC_REPO_COLS].copy()
            for col in NUMERIC_REPO_COLS:
                numerics[col] = (numerics[col].astype(float) - means[col]) / stds[col]
            numerics_tensor = torch.tensor(
                numerics.values, dtype=torch.float32, device=device
            )

            user_tensor = u_tensor.expand(len(group))

            logits, _, _ = model(user_tensor, item_indices, lang_indices, numerics_tensor)
            probs = torch.sigmoid(logits).detach().cpu().numpy()

            ranked_idx = np.argsort(-probs)
            topk_idx = ranked_idx[:k]

            labels_arr = labels.astype(int)
            hit = int(labels_arr[topk_idx].max() == 1)
            hits.append(hit)

    if len(hits) == 0:
        return float("nan")
    return float(np.mean(hits))

########################################
# 6. MAIN SCRIPT
########################################

def main():
    cfg = Config()
    set_seed(cfg.seed)

    print("Loading raw data...")
    train_bal, train_neg, test_bal, test_neg = load_raw_data(cfg)

    # Combine training pool and test set
    train_pool = pd.concat([train_bal, train_neg], ignore_index=True)
    test_df = pd.concat([test_bal, test_neg], ignore_index=True)

    print(f"Train pool size: {len(train_pool)}, Test size: {len(test_df)}")

    print("Splitting train pool into train and validation by user...")
    train_df, val_df = split_train_val_by_user(
        train_pool,
        train_user_fraction=cfg.train_user_fraction,
        seed=cfg.seed,
    )

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    print("Building ID mappings (users/items/langs)...")
    user2idx, item2idx, lang2idx, idx2user, idx2item, idx2lang = build_id_mappings(
        train_pool, test_df
    )
    num_users = len(user2idx)
    num_items = len(item2idx)
    num_langs = len(lang2idx)

    print(f"Num users: {num_users}, num items: {num_items}, num langs: {num_langs}")

    print("Fitting numeric scalers on training pool...")
    means, stds = compute_numeric_scalers(train_pool)

    print("Normalizing numeric features...")
    train_df_norm = normalize_numeric_features(train_df, means, stds)
    val_df_norm = normalize_numeric_features(val_df, means, stds)
    test_df_norm = normalize_numeric_features(test_df, means, stds)

    print("Creating dataloaders for train and validation...")
    train_loader, val_loader = make_dataloaders(
        cfg, train_df_norm, val_df_norm, user2idx, item2idx, lang2idx
    )

    print("Initializing model...")
    model = TwoTowerRecSys(
        num_users=num_users,
        num_items=num_items,
        num_langs=num_langs,
        num_numeric_feats=len(NUMERIC_REPO_COLS),
        cfg=cfg,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_auc = -1.0

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, cfg.device)
        print(f"Train loss: {train_loss:.4f}")

        val_loss, val_auc = evaluate_pointwise(model, val_loader, cfg.device)
        print(f"Val loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

        if SKLEARN_AVAILABLE and not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_twotower_model.pt")
            print("Saved new best model based on validation AUC.")

    # Load best model (if saved)
    if os.path.exists("best_twotower_model.pt"):
        model.load_state_dict(torch.load("best_twotower_model.pt", map_location=cfg.device))
        print("Loaded best model from checkpoint.")

    # Final evaluation on test set
    print("\nCreating test DataLoader for pointwise metrics...")
    test_dataset = TwoTowerDataset(test_df_norm, user2idx, item2idx, lang2idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loss, test_auc = evaluate_pointwise(model, test_loader, cfg.device)
    print(f"Test loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")

    print("\nEvaluating Hit@5 on test set...")
    hit_at_5 = evaluate_hit_at_k(
        model,
        test_df,
        user2idx,
        item2idx,
        lang2idx,
        means,
        stds,
        cfg.device,
        k=5,
    )
    print(f"Test Hit@5: {hit_at_5:.4f}")

if __name__ == "__main__":
    main()