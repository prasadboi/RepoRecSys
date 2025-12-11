"""
Quick script to find a valid user ID from the artifacts
"""
import pickle

# Load artifacts
with open('model_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

# Get some valid user IDs
user2idx = artifacts['user2idx']
idx2user = artifacts['idx2user']

print(f"Total users in training data: {len(user2idx)}")
print("\nFirst 10 valid user IDs:")
for i in range(min(10, len(user2idx))):
    user_id = idx2user[i]
    print(f"  User ID: {user_id}")

print(f"\nTry testing with user_id: {idx2user[0]}")

