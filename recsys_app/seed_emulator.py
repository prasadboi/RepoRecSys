import os
from google.cloud import storage

# Point to the local emulator
# Before:
# os.environ["STORAGE_EMULATOR_HOST"] = "http://localhost:4443"

# After:
os.environ.setdefault("STORAGE_EMULATOR_HOST", "http://localhost:4443")

# Config
BUCKET_NAME = "recsys-data-bucket"
SOURCE_DIR = "artifacts"
DEST_PREFIX = "latest_model"

def seed_bucket():
    # 1. Initialize Client (points to localhost:4443 due to env var)
    client = storage.Client()

    # 2. Create Bucket if not exists
    try:
        bucket = client.create_bucket(BUCKET_NAME)
        print(f"Created bucket: {BUCKET_NAME}")
    except Exception:
        bucket = client.bucket(BUCKET_NAME)
        print(f"Bucket {BUCKET_NAME} already exists")

    # 3. Upload Files
    files = [
        "best_twotower_model.pt", 
        "mappings.pkl", 
        "scalers.pkl", 
        "item_features.csv",
        "train_balanced.csv"  # <--- Add this
    ]

    for filename in files:
        local_path = os.path.join(SOURCE_DIR, filename)
        if not os.path.exists(local_path):
            print(f"Skipping {filename} (Not found in {SOURCE_DIR})")
            continue

        blob_path = f"{DEST_PREFIX}/{filename}"
        if filename == "train_balanced.csv":
            blob_path = "archive/base_train.csv"

        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {filename} -> gs://{BUCKET_NAME}/{blob_path}")

if __name__ == "__main__":
    seed_bucket()