#!/bin/bash
# Set up API as a systemd service (auto-starts on boot, auto-restarts on crash)

cd ~/github-recommender

# Create systemd service file
sudo tee /etc/systemd/system/github-recommender.service > /dev/null <<EOF
[Unit]
Description=GitHub Repository Recommender API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/github-recommender
Environment="PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/usr/bin/python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable github-recommender
sudo systemctl start github-recommender

echo "Service installed and started!"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status github-recommender  # Check status"
echo "  sudo systemctl restart github-recommender # Restart service"
echo "  sudo systemctl stop github-recommender     # Stop service"
echo "  sudo journalctl -u github-recommender -f  # View logs"

