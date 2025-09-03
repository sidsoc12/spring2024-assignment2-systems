#!/bin/bash
# A robust setup script for a CLEAN Ubuntu 22.04 Lambda Labs instance.

set -e

echo "--- [1/6] Updating system packages ---"
sudo apt-get update
sudo apt-get install -y git wget

echo ""
echo "--- [2/6] Installing NVIDIA CUDA Toolkit ---"
# This is the official NVIDIA process for a clean Ubuntu 22.04 install.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
# Install the full toolkit, which guarantees nsys and other tools are present.
sudo apt-get install -y cuda-toolkit-12-2

echo ""
echo "--- [3/6] Configuring PATH for NVIDIA tools ---"
# Add the CUDA bin directory to our PATH for this session and future ones.
echo 'export PATH="/usr/local/cuda-12.2/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

echo ""
echo "--- [4/6] Installing uv package manager ---"
curl -LsSf https://astral.sh/uv/install.sh | sh
# Add uv to our PATH for this session and future ones.
echo 'export PATH="/home/ubuntu/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

echo ""
echo "--- [5/6] Cloning your assignment repository ---"
cd /home/ubuntu
git clone https://github.com/sidsoc12/spring2024-assignment2-systems.git

echo ""
echo "--- [6/6] Verifying installations ---"
echo "Verifying NVIDIA tools..."
nvidia-smi
echo "Verifying nsys..."
nsys --version
echo "Verifying uv..."
uv --version

echo ""
echo "--- ✅ Setup Complete! ---"
echo "Your new instance is fully configured and ready to go."
echo "You can now 'cd spring2024-assignment2-systems' and run your profiler."