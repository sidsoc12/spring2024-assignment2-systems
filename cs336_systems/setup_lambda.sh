#!/bin/bash

# This script automates the setup of a fresh Lambda Labs instance for the cs336 assignment.
# It performs the following steps:
# 1. Updates the system's package list.
# 2. Installs git.
# 3. Downloads and installs the 'uv' package manager.
# 4. Permanently adds 'uv' to the system's PATH.
# 5. Clones your specific assignment repository from GitHub.

# --- Start of Script ---

# Use -e to exit immediately if any command fails
set -e

echo "--- [Step 1/5] Updating package lists with apt-get... ---"
sudo apt-get update

echo ""
echo "--- [Step 2/5] Installing git... ---"
sudo apt-get install git -y

echo ""
echo "--- [Step 3/5] Installing uv package manager... ---"
# Download and run the official uv installation script
curl -LsSf https://astral.sh/uv/install.sh | sh

echo ""
echo "--- [Step 4/5] Configuring shell to find the uv command... ---"
# The uv installer places the binary in /home/ubuntu/.local/bin
# We need to add this directory to the PATH environment variable.
# This command adds the necessary line to the .bashrc file, which runs every time you log in.
echo 'export PATH="/home/ubuntu/.local/bin:$PATH"' >> ~/.bashrc

# Apply the change to the current terminal session immediately
source ~/.bashrc

echo ""
echo "--- [Step 5/5] Cloning your assignment repository from GitHub... ---"
# Change to the home directory to ensure the repo is cloned in the right place
cd /home/ubuntu
# Clone your specific forked repository
git clone https://github.com/sidsoc12/spring2024-assignment2-systems.git

echo ""
echo "--- ✅ Setup Complete! ---"
echo "Git and uv are installed, and your repository has been cloned."
echo "You can now 'cd spring2024-assignment2-systems' and run your script."

