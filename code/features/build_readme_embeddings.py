"""
Build README embeddings for repos in the recsys dataset using BigQuery
and BERT-style embeddings.

Stages:
  1. Collect unique project_ids from all train/test CSVs and join with repo_name.
     Save locally and upload to BigQuery as a small table.

  2. In BigQuery, join project_ids table with github_repos.files/contents
     to materialize a readmes table (one-time expensive query).

  3. Download the readmes table to a local file.

  4. Compute text embeddings for each README and save them.

You can toggle which stages run using the RUN_STAGE_* flags below.
"""

import os
from dataclasses import dataclass
from typing import List

import pandas as pd
from google.cloud import bigquery  # pip install google-cloud-bigquery
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
import numpy as np

# -------------------
# Configuration
# -------------------

@dataclass
class Config:
    # Local CSVs from your recsys project
    train_balanced_path: str = "/Users/prasadboi/Work/RepoRecSys/data/train_balanced.csv"
    train_negative_path: str = "/Users/prasadboi/Work/RepoRecSys/data/train_negative.csv"
    test_balanced_path: str = "/Users/prasadboi/Work/RepoRecSys/data/test_balanced.csv"
    test_negative_path: str = "/Users/prasadboi/Work/RepoRecSys/data/test_negative.csv"

    # Mapping: project_id -> repo_name ("owner/repo")
    # You need to create this once if you don’t already have it.
    project_repo_map_path: str = "/Users/prasadboi/Work/RepoRecSys/data/features/repo_id_to_repo_name_map.csv"

    # Where to store intermediate artifacts locally
    out_dir: str = "/Users/prasadboi/Work/RepoRecSys/data/features"
    unique_projects_csv: str = "/Users/prasadboi/Work/RepoRecSys/data/features/unique_ids.csv"
    mapped_projects_csv: str = "/Users/prasadboi/Work/RepoRecSys/data/features/project_repo_mapped.csv"
    readmes_local_parquet: str = "/Users/prasadboi/Work/RepoRecSys/data/features/project_readmes.parquet"
    embeddings_npy: str = "/Users/prasadboi/Work/RepoRecSys/data/features/project_readme_embeddings.npy"
    embeddings_index_csv: str = "/Users/prasadboi/Work/RepoRecSys/data/features/project_readme_embeddings_index.csv"

    # BigQuery settings
    gcp_project_id: str = "cloudandml-471822"       # <- set this
    bq_dataset: str = "recsys_repo_text"              # dataset in your project
    bq_project_ids_table: str = "project_ids"         # will create <dataset>.project_ids
    bq_readmes_table: str = "project_readmes"         # will create <dataset>.project_readmes

    # BERT / sentence-transformer model
    text_model_name: str = "all-MiniLM-L6-v2"  # small but good; swap if you want a vanilla BERT

cfg = Config()

# Toggle stages here
RUN_STAGE_1 = False   # collect unique project_ids + upload to BQ
RUN_STAGE_2 = True   # BQ query to materialize READMEs table (one-time)
RUN_STAGE_3 = True  # download READMEs from BQ to local parquet
RUN_STAGE_4 = False   # compute embeddings and save locally

PROJECT_ID_COL = "project_id"
REPO_NAME_COL = "repo_name"
README_CONTENT_COL = "content"


# -------------------
# Helpers
# -------------------

def ensure_out_dir():
    os.makedirs(cfg.out_dir, exist_ok=True)


def get_bq_client() -> bigquery.Client:
    """
    Create a BigQuery client using Application Default Credentials.
    You can auth via `gcloud auth application-default login`
    or Colab's `from google.colab import auth; auth.authenticate_user()`.
    :contentReference[oaicite:1]{index=1}
    """
    return bigquery.Client(project=cfg.gcp_project_id)


# -------------------
# Stage 1:
# Collect unique project_ids and upload to BigQuery
# -------------------

def stage1_collect_and_upload_project_ids():
    print("=== Stage 1: Collect unique project_ids and upload to BigQuery ===")
    ensure_out_dir()

    # 1a. Read all four recsys CSVs
    print("Reading recsys CSVs...")
    dfs: List[pd.DataFrame] = []
    for path in [
        cfg.train_balanced_path,
        cfg.train_negative_path,
        cfg.test_balanced_path,
        cfg.test_negative_path,
    ]:
        df = pd.read_csv(path, usecols=[PROJECT_ID_COL])
        dfs.append(df)

    all_ids = pd.concat(dfs, ignore_index=True)
    print(f"Total rows across all files: {len(all_ids):,}")

    # 1b. Get unique project_ids
    unique_ids = (
        all_ids[PROJECT_ID_COL]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    print(f"Unique project_ids: {len(unique_ids):,}")

    unique_df = pd.DataFrame({PROJECT_ID_COL: unique_ids})

    # 1c. Join with project_repo_map to get repo_name
    if not os.path.exists(cfg.project_repo_map_path):
        raise FileNotFoundError(
            f"Expected mapping file at {cfg.project_repo_map_path} with columns "
            f"{PROJECT_ID_COL}, {REPO_NAME_COL}"
        )

    map_df = pd.read_csv(cfg.project_repo_map_path)
    if REPO_NAME_COL not in map_df.columns:
        raise ValueError(
            f"{cfg.project_repo_map_path} must contain a '{REPO_NAME_COL}' column "
            "with GitHub repo names like 'owner/repo'."
        )

    mapped = unique_df.merge(map_df[[PROJECT_ID_COL, REPO_NAME_COL]],
                             on=PROJECT_ID_COL,
                             how="left")

    missing = mapped[mapped[REPO_NAME_COL].isna()]
    if len(missing) > 0:
        print(f"WARNING: {len(missing):,} project_ids have no repo_name mapping. "
              "They will be dropped for README extraction.")
        mapped = mapped.dropna(subset=[REPO_NAME_COL])

    print(f"Mappable project_ids: {len(mapped):,}")

    # 1d. Save locally
    mapped_path = os.path.join(cfg.out_dir, cfg.mapped_projects_csv)
    mapped.to_csv(mapped_path, index=False)
    print(f"Saved project_id -> repo_name mapping to {mapped_path}")

    # 1e. Upload to BigQuery as a small table
    client = get_bq_client()
    table_id = f"{cfg.gcp_project_id}.{cfg.bq_dataset}.{cfg.bq_project_ids_table}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
    )
    print(f"Uploading mapping to BigQuery table {table_id} ...")
    with open(mapped_path, "rb") as f:
        load_job = client.load_table_from_file(f, table_id, job_config=job_config)
    load_job.result()
    print("Upload complete.")


# -------------------
# Stage 2:
# BigQuery query to materialize README table
# -------------------

def stage2_query_readmes():
    print("=== Stage 2: BigQuery README extraction ===")
    client = get_bq_client()

    # Target table we will create or overwrite once.
    readmes_table_id = f"{cfg.gcp_project_id}.{cfg.bq_dataset}.{cfg.bq_readmes_table}"
    project_ids_table_id = f"{cfg.gcp_project_id}.{cfg.bq_dataset}.{cfg.bq_project_ids_table}"

    # We join our small project_ids table to the large GitHub tables once.
    # This query now:
    #   - finds README files at root or in subdirs (docs/README.md, src/README, etc.)
    #   - matches README, README.md, README.rst, etc.
    #   - keeps only non-binary text files
    query = f"""
    CREATE OR REPLACE TABLE `{readmes_table_id}` AS
    SELECT
      m.{PROJECT_ID_COL} AS {PROJECT_ID_COL},
      f.repo_name AS {REPO_NAME_COL},
      f.path,
      c.content AS {README_CONTENT_COL}
    FROM `{project_ids_table_id}` AS m
    JOIN `bigquery-public-data.github_repos.files` AS f
      ON m.{REPO_NAME_COL} = f.repo_name
    JOIN `bigquery-public-data.github_repos.contents` AS c
      ON f.id = c.id
    WHERE
      c.binary = FALSE
      AND REGEXP_CONTAINS(
        LOWER(f.path),
        r'(^|/)readme(\\.[a-z0-9]+)?$'
      )
    """

    # Reasoning:
    # - We SELECT from github_repos.files/contents once, scanning them fully only this time.
    # - Future runs just read from {readmes_table_id}, which is small and cheap.
    # - REGEXP_CONTAINS lets us match README-like filenames anywhere in the path.

    print(f"Running README extraction query; results go to {readmes_table_id}")
    job = client.query(query)
    job.result()
    print("BigQuery README table created/refreshed.")


# -------------------
# Stage 3:
# Download README table to local file
# -------------------

def stage3_download_readmes():
    print("=== Stage 3: Download READMEs from BigQuery ===")
    ensure_out_dir()
    client = get_bq_client()

    readmes_table_id = f"{cfg.gcp_project_id}.{cfg.bq_dataset}.{cfg.bq_readmes_table}"

    query = f"""
    SELECT {PROJECT_ID_COL}, {REPO_NAME_COL}, path, {README_CONTENT_COL}
    FROM `{readmes_table_id}`
    """

    print("Querying readmes table into Pandas DataFrame...")
    df = client.query(query).to_dataframe()  # documented pattern :contentReference[oaicite:4]{index=4}
    print(f"Downloaded {len(df):,} README rows from BigQuery.")

    out_path = os.path.join(cfg.out_dir, cfg.readmes_local_parquet)
    df.to_parquet(out_path, index=False)
    print(f"Saved READMEs to {out_path}")


# -------------------
# Stage 4:
# Compute BERT-style embeddings for READMEs
# -------------------

def stage4_compute_embeddings():
    print("=== Stage 4: Compute text embeddings ===")
    ensure_out_dir()

    readmes_path = os.path.join(cfg.out_dir, cfg.readmes_local_parquet)
    if not os.path.exists(readmes_path):
        raise FileNotFoundError(
            f"{readmes_path} not found. Run Stage 3 first (download READMEs)."
        )

    df = pd.read_parquet(readmes_path)
    print(f"Loaded {len(df):,} README rows for embedding.")

    # Clean up text a bit (optional)
    texts = df[README_CONTENT_COL].fillna("").astype(str).tolist()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load sentence-transformer (uses a BERT-like encoder with mean pooling).
    # This is much better for similarity than raw BERT CLS embeddings. :contentReference[oaicite:5]{index=5}
    model = SentenceTransformer(cfg.text_model_name)

    batch_size = 64
    embeddings = []
    n = len(texts)

    # We use encode with show_progress_bar to avoid keeping everything in memory at once.
    print(f"Encoding {n:,} READMEs in batches of {batch_size} ...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # embeddings shape: [N, D]
    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings and an index file with project_id + row index
    emb_path = os.path.join(cfg.out_dir, cfg.embeddings_npy)
    np.save(emb_path, embeddings)
    print(f"Saved embedding matrix to {emb_path}")

    index_df = df[[PROJECT_ID_COL, REPO_NAME_COL]].reset_index().rename(
        columns={"index": "row_idx"}
    )
    index_path = os.path.join(cfg.out_dir, cfg.embeddings_index_csv)
    index_df.to_csv(index_path, index=False)
    print(f"Saved embedding index (project_id -> row_idx) to {index_path}")

    print("Stage 4 complete.")


# -------------------
# Main
# -------------------

if __name__ == "__main__":
    if RUN_STAGE_1:
        stage1_collect_and_upload_project_ids()

    if RUN_STAGE_2:
        stage2_query_readmes()

    if RUN_STAGE_3:
        stage3_download_readmes()

    if RUN_STAGE_4:
        stage4_compute_embeddings()
