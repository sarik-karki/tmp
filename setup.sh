#!/bin/bash
set -e

echo "=== Parking Monitor Setup ==="

# System dependencies
echo "Installing system packages..."
sudo apt update
sudo apt install -y tesseract-ocr libatlas-base-dev python3-pip python3-venv

# Python virtual environment
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages (use piwheels for prebuilt ARM wheels)
echo "Installing Python packages..."
pip install --upgrade pip
pip install --extra-index-url https://www.piwheels.org/simple -r requirements.txt

# Create data directory
mkdir -p data

echo ""
echo "=== Setup complete ==="
echo "To run:"
echo "  source venv/bin/activate"
echo "  python3 main.py"
