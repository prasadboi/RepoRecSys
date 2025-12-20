#!/usr/bin/env python3
"""
extract_unique_repo_ids.py

Reads all recsys interaction CSVs (train/test, balanced/negative),
collects the unique repo IDs (project_id), and writes them to a CSV.

Usage (with your current paths):

    python extract_unique_repo_ids.py \
        --train-balanced "/content/drive/MyDrive/Project_Work/RepoRecSys/data/train_balanced.csv" \
        --train-negative "/content/drive/MyDrive/Project_Work/RepoRecSys/data/train_negative.csv" \
        --test-balanced  "/content/drive/MyDrive/Project_Work/RepoRecSys/data/test_balanced.csv" \
        --test-negative  "/content/drive/MyDrive/Project_Work/RepoRecSys/data/test_negative.csv" \
        --output "/content/drive/MyDrive/Project_Work/RepoRecSys/data/unique_repo_ids.csv"

You can change the paths via CLI flags as needed.
"""

import argparse
import os
from typing import List

import pandas as pd


PROJECT_ID_COL = "project_id"   # this is the repo_id in your dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique repo IDs (project_id) from recsys CSVs."
    )
    parser.add_argument(
        "--train-balanced",
        required=True,
        help="Path to train_balanced.csv",
    )
    parser.add_argument(
        "--train-negative",
        required=True,
        help="Path to train_negative.csv",
    )
    parser.add_argument(
        "--test-balanced",
        required=True,
        help="Path to test_balanced.csv",
    )
    parser.add_argument(
        "--test-negative",
        required=True,
        help="Path to test_negative.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV of unique repo IDs",
    )
    return parser.parse_args()


def read_project_ids(path: str) -> pd.Series:
    """
    Read a single CSV and return the project_id column as a pandas Series.

    We only load the one column we need to save memory and speed things up.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    print(f"Reading {path} ...")
    df = pd.read_csv(path, usecols=[PROJECT_ID_COL])
    if PROJECT_ID_COL not in df.columns:
        raise ValueError(
            f"{path} does not contain required column '{PROJECT_ID_COL}'"
        )

    return df[PROJECT_ID_COL]


def main():
    args = parse_args()

    # 1. Read project_id column from each file
    series_list: List[pd.Series] = []
    for p in [
        args.train_balanced,
        args.train_negative,
        args.test_balanced,
        args.test_negative,
    ]:
        s = read_project_ids(p)
        series_list.append(s)

    # 2. Concatenate and deduplicate
    all_ids = pd.concat(series_list, ignore_index=True)
    print(f"Total rows across all files: {len(all_ids):,}")

    # Drop NaN just in case, and cast to int (your IDs are numeric)
    all_ids = all_ids.dropna()
    # If some IDs might be non-integer, you can skip the astype(int) cast.
    all_ids = all_ids.astype("int64")

    unique_ids = (
        all_ids.drop_duplicates().sort_values().reset_index(drop=True)
    )
    print(f"Unique repo IDs (project_id): {len(unique_ids):,}")

    # 3. Save to CSV
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df = pd.DataFrame({PROJECT_ID_COL: unique_ids})
    out_df.to_csv(args.output, index=False)
    print(f"Saved unique repo IDs to: {args.output}")


if __name__ == "__main__":
    main()
