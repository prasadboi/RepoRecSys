#!/bin/bash
# Update model files without redeploying
# Upload new model files, then run this script

cd ~/github-recommender

echo "Updating model files..."

# Check if new files exist
if [ ! -f "best_twotower_model.pt" ]; then
    echo "Error: best_twotower_model.pt not found in current directory"
    exit 1
fi

if [ ! -f "model_artifacts.pkl" ]; then
    echo "Error: model_artifacts.pkl not found in current directory"
    exit 1
fi

if [ ! -f "item_metadata.pkl" ]; then
    echo "Warning: item_metadata.pkl not found, continuing anyway..."
fi

# Backup old files
echo "Backing up old files..."
mkdir -p backups
timestamp=$(date +%Y%m%d_%H%M%S)
[ -f "best_twotower_model.pt" ] && cp best_twotower_model.pt backups/best_twotower_model.pt.$timestamp
[ -f "model_artifacts.pkl" ] && cp model_artifacts.pkl backups/model_artifacts.pkl.$timestamp
[ -f "item_metadata.pkl" ] && cp item_metadata.pkl backups/item_metadata.pkl.$timestamp

echo "Files backed up to backups/"

# Restart service to load new model
if systemctl is-active --quiet github-recommender; then
    echo "Restarting service to load new model..."
    sudo systemctl restart github-recommender
    sleep 3
    sudo systemctl status github-recommender --no-pager
else
    echo "Service not running. Start it with: ./start_api.sh or sudo systemctl start github-recommender"
fi

echo ""
echo "Update complete! Model should be reloaded."

