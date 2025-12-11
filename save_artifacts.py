"""
Script to save model artifacts needed for inference
Run this after training to save mappings and scalers
"""
import pickle
import pandas as pd

# Try to import from main_colab (for Colab) or main (for local)
try:
    from main_colab import (
        Config, load_raw_data, build_id_mappings, 
        compute_numeric_scalers, USER_ID_COL, ITEM_ID_COL, NUMERIC_REPO_COLS
    )
except ImportError:
    from main import (
        Config, load_raw_data, build_id_mappings, 
        compute_numeric_scalers, USER_ID_COL, ITEM_ID_COL, NUMERIC_REPO_COLS
    )

def save_artifacts():
    cfg = Config()
    
    print("Loading data to build mappings...")
    train_bal, train_neg, test_bal, test_neg = load_raw_data(cfg)
    train_pool = pd.concat([train_bal, train_neg], ignore_index=True)
    test_df = pd.concat([test_bal, test_neg], ignore_index=True)
    
    print("Building ID mappings...")
    user2idx, item2idx, lang2idx, idx2user, idx2item, idx2lang = build_id_mappings(
        train_pool, test_df
    )
    
    print("Computing scalers...")
    means, stds = compute_numeric_scalers(train_pool)
    
    # Create project_id to GitHub URL mapping
    # Assuming project_id might be GitHub repo ID or we need to construct URL
    # For now, we'll create a placeholder - you may need to adjust based on your data
    project_id_to_url = {}
    all_items = pd.concat([train_pool[ITEM_ID_COL], test_df[ITEM_ID_COL]]).unique()
    
    # Try to get repo info from training data if available
    # If your data has repo names/owners, use those to construct URLs
    # For now, creating a mapping that assumes project_id format
    for item_id in all_items:
        # This is a placeholder - adjust based on your actual data structure
        # You might need to query GitHub API or have a separate mapping file
        project_id_to_url[item_id] = f"https://github.com/repo/{item_id}"
    
    # Save all artifacts
    artifacts = {
        'user2idx': user2idx,
        'item2idx': item2idx,
        'lang2idx': lang2idx,
        'idx2user': idx2user,
        'idx2item': idx2item,
        'idx2lang': idx2lang,
        'means': means,
        'stds': stds,
        'project_id_to_url': project_id_to_url,
        'num_users': len(user2idx),
        'num_items': len(item2idx),
        'num_langs': len(lang2idx),
        'num_numeric_feats': len(NUMERIC_REPO_COLS),
        'config': {
            'user_id_emb_dim': cfg.user_id_emb_dim,
            'item_id_emb_dim': cfg.item_id_emb_dim,
            'lang_emb_dim': cfg.lang_emb_dim,
            'hidden_dim': cfg.hidden_dim,
            'embedding_dim': cfg.embedding_dim,
        }
    }
    
    with open('model_artifacts.pkl', 'wb') as f:
        pickle.dump(artifacts, f)
    
    print("Saved artifacts to model_artifacts.pkl")
    print(f"Num users: {len(user2idx)}, Num items: {len(item2idx)}, Num langs: {len(lang2idx)}")
    
    # Also save item metadata for easier lookup
    # Try to extract repo info from test data if available
    item_metadata = {}
    if ITEM_ID_COL in test_df.columns:
        # Group by item_id and get first row for each item
        grouped = test_df.groupby(ITEM_ID_COL).first()
        for item_id, row in grouped.iterrows():
            item_metadata[item_id] = {
                'language_code': row.get('language_code', 'unknown'),
                'watchers': int(row.get('watchers', 0)),
                'commits': int(row.get('commits', 0)),
            }
    else:
        # If column doesn't exist, create empty metadata
        print(f"Warning: {ITEM_ID_COL} column not found in test_df. Creating empty metadata.")
        for item_id in all_items:
            item_metadata[item_id] = {
                'language_code': 'unknown',
                'watchers': 0,
                'commits': 0,
            }
    
    with open('item_metadata.pkl', 'wb') as f:
        pickle.dump(item_metadata, f)
    
    print(f"Saved item metadata to item_metadata.pkl ({len(item_metadata)} items)")

if __name__ == "__main__":
    save_artifacts()

