#!/bin/bash
# Install dependencies locally

echo "Installing Python dependencies..."
pip3 install -r requirements.txt

echo ""
echo "Installation complete!"
echo ""
echo "To verify, run: python3 -c 'import torch; print(f\"PyTorch version: {torch.__version__}\")'"

