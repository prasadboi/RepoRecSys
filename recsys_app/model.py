import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset
from typing import Dict

# Define columns here to ensure the Dataset class knows which to read
NUMERIC_COLS = [
    "watchers", "commits", "issues", "pull_requests",
    "mean_commits_language", "max_commits_language", "min_commits_language", "std_commits_language",
    "mean_pull_requests_language", "max_pull_requests_language", "min_pull_requests_language", "std_pull_requests_language",
    "mean_issues_language", "max_issues_language", "min_issues_language", "std_issues_language",
    "mean_watchers_language", "max_watchers_language", "min_watchers_language", "std_watchers_language",
    "events", "year", "weight", "cp", "avg_cp", "stddev",
]

class Config:
    user_id_emb_dim = 64
    item_id_emb_dim = 64
    lang_emb_dim = 16
    hidden_dim = 128
    embedding_dim = 64
    dropout = 0.0
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        self.user_ids = self.df["id_user"].values
        self.item_ids = self.df["project_id"].values
        self.lang_codes = self.df["language_code"].values
        self.labels = self.df["target"].astype(float).values
        
        # Ensure we only use the valid numeric columns
        self.numeric_matrix = self.df[NUMERIC_COLS].astype(float).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        uid = self.user_ids[idx]
        iid = self.item_ids[idx]
        lang = self.lang_codes[idx]
        
        # Note: In training loop we don't use the label (it's implicit 1s), 
        # but we return it to keep signature consistent
        label = self.labels[idx]
        numerics = self.numeric_matrix[idx]

        u_idx = self.user2idx.get(uid, 0) # Fallback to 0 if unknown (rare in train)
        i_idx = self.item2idx.get(iid, 0)
        l_idx = self.lang2idx.get(lang, 0)

        # Convert to tensors
        u_idx = torch.tensor(u_idx, dtype=torch.long)
        i_idx = torch.tensor(i_idx, dtype=torch.long)
        l_idx = torch.tensor(l_idx, dtype=torch.long)
        numerics = torch.tensor(numerics, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return u_idx, i_idx, l_idx, numerics, label

class UserTower(nn.Module):
    def __init__(self, num_users: int, cfg: Config):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, cfg.user_id_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.user_id_emb_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )

    def forward(self, user_ids: torch.Tensor):
        x = self.user_emb(user_ids)
        x = self.mlp(x)
        return nn.functional.normalize(x, p=2, dim=-1)

class ItemTower(nn.Module):
    def __init__(self, num_items: int, num_langs: int, num_numeric_feats: int, cfg: Config):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, cfg.item_id_emb_dim)
        self.lang_emb = nn.Embedding(num_langs, cfg.lang_emb_dim)
        
        self.num_mlp = nn.Sequential(
            nn.Linear(num_numeric_feats, 64),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        fused_dim = cfg.item_id_emb_dim + cfg.lang_emb_dim + 32
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.embedding_dim),
        )

    def forward(self, item_ids, lang_ids, numeric_feats):
        item_e = self.item_emb(item_ids)
        lang_e = self.lang_emb(lang_ids)
        num_e = self.num_mlp(numeric_feats)
        x = torch.cat([item_e, lang_e, num_e], dim=-1)
        x = self.mlp(x)
        return nn.functional.normalize(x, p=2, dim=-1)

class TwoTowerRecSys(nn.Module):
    def __init__(self, num_users, num_items, num_langs, num_numeric_feats, cfg):
        super().__init__()
        self.user_tower = UserTower(num_users, cfg)
        self.item_tower = ItemTower(num_items, num_langs, num_numeric_feats, cfg)

    def forward(self, user_ids, item_ids, lang_ids, numeric_feats):
        u = self.user_tower(user_ids)
        v = self.item_tower(item_ids, lang_ids, numeric_feats)
        return torch.sum(u * v, dim=-1), u, v