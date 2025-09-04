#!/bin/bash
# The final, definitive setup script for a LAMBDA STACK 22.04 instance.

set -e # Exit immediately if any command fails

echo "--- [1/5] Installing prerequisite tools (git, uv, nsys) ---"
sudo apt-get update
sudo apt-get install -y git wget
sudo snap install astral-uv --classic

# This is the new, correct command to install a working version of Nsight Systems
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_5/NsightSystems-linux-cli-public-2025.5.1.121-3638078.deb && \
sudo apt install -y ./NsightSystems-linux-cli-public-2025.5.1.121-3638078.deb

echo ""
echo "--- [2/5] Cloning your assignment repository ---"
cd /home/ubuntu
# Your default branch is main, so this will clone the correct code
git clone https://github.com/sidsoc12/spring2024-assignment2-systems.git

echo ""
echo "--- [3/5] Changing directory and fixing dependencies ---"
cd spring2024-assignment2-systems
# Use sed to replace the 'tabulate' dependency with a specific, working version.
sed -i 's/"tabulate",/"tabulate==0.9.0",/' pyproject.toml
echo "pyproject.toml has been updated."

echo ""
echo "--- [4/5] Verifying installations ---"
echo "Verifying nsys..."
nsys --version
echo "Verifying uv..."
uv --version

echo ""
echo "--- [5/5] ✅ Setup Complete! ---"
echo "You are now inside the project directory and can run your profiler."