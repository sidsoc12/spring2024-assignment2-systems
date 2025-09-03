#!/bin/bash
# A targeted setup script for the LAMBDA STACK 22.04 image.

set -e

echo "--- [1/4] Installing git and uv ---"
sudo apt-get update
sudo apt-get install -y git
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="/home/ubuntu/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

echo ""
echo "--- [2/4] Cloning your repository ---"
cd /home/ubuntu
git clone https://github.com/sidsoc12/spring2024-assignment2-systems.git

echo ""
echo "--- [3/4] DEFINITIVE NSYS FIX: Reconfiguring NVIDIA's repository ---"
# Clean up any broken Lambda Stack configurations
sudo rm -f /etc/apt/sources.list.d/cuda*
# Download and install the official NVIDIA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
# Force an update to read from the correct new source
sudo apt-get update

echo ""
echo "--- [4/4] Installing a stable version of Nsight Systems ---"
sudo apt-get install -y cuda-nsight-systems-12-2

echo ""
echo "--- ✅ Setup Complete! ---"
echo "Verifying nsys installation..."
nsys --version
echo "You can now 'cd spring2024-assignment2-systems' and run your profiler."