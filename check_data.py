"""
Helper script to check if data files exist and show their locations
"""
import os
from main import Config

def check_data_files():
    cfg = Config()
    
    print("Checking for data files...")
    print(f"Data directory: {cfg.data_dir}")
    print()
    
    files_to_check = [
        ("train_balanced.csv", cfg.train_balanced_path),
        ("train_negative.csv", cfg.train_negative_path),
        ("test_balanced.csv", cfg.test_balanced_path),
        ("test_negative.csv", cfg.test_negative_path),
    ]
    
    all_exist = True
    for name, path in files_to_check:
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_exist = False
    
    print()
    if all_exist:
        print("✓ All data files found! You can run main.py")
    else:
        print("✗ Some files are missing.")
        print("\nTo fix:")
        print(f"1. Create directory: mkdir -p {cfg.data_dir}")
        print("2. Place your CSV files in that directory")
        print("3. Or update Config.data_dir in main.py to point to your data location")

if __name__ == "__main__":
    check_data_files()

