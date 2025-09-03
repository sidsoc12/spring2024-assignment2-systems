#!/bin/bash

# This script automates the setup of a fresh Lambda Labs instance for the cs336 assignment.
# It installs git, uv, Nsight Systems, and clones your repository.

# Use -e to exit immediately if any command fails
set -e

echo "--- [Step 1/6] Updating package lists with apt-get... ---"
sudo apt-get update

echo ""
echo "--- [Step 2/6] Installing git... ---"
sudo apt-get install git -y

echo ""
echo "--- [Step 3/6] Installing uv package manager... ---"
curl -LsSf https://astral.sh/uv/install.sh | sh

echo ""
echo "--- [Step 4/6] Configuring shell to find the uv command... ---"
echo 'export PATH="/home/ubuntu/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

echo ""
echo "--- [Step 5/6] Installing Nsight Systems Profiler... ---"
# Installs the specific version of Nsight Systems that we found.
sudo apt-get install -y nsight-systems-2025.3.2

echo ""
echo "--- [Step 6/6] Cloning repository and changing directory... ---"
cd /home/ubuntu
git clone https://github.com/sidsoc12/spring2024-assignment2-systems.git
# CD into the newly cloned repository directory.
cd spring2024-assignment2-systems

echo ""
echo "--- ✅ Setup Complete! ---"
echo "All tools are installed and you are now inside the project directory."
echo "You can run your profiler command directly."