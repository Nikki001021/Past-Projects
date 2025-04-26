#!/bin/bash
set -e  # Exit immediately if any command fails

echo "Recombining and extracting DoTA and BDD frame datasets..."

# Set working directory
WORK_DIR=$PWD

# Combine parts
cat DoTA_Frames.tar.gz.part_* > "${WORK_DIR}/DoTA_Frames.tar.gz"
cat BDD_Frames.tar.gz.part_* > "${WORK_DIR}/BDD_Frames.tar.gz"

# Make sure target folders exist
mkdir -p "${WORK_DIR}/DoTA/DoTA_Frames"
mkdir -p "${WORK_DIR}/BDD100K/BDD_Frames"

# Extract into the right folders
tar -xzvf "${WORK_DIR}/DoTA_Frames.tar.gz" -C "${WORK_DIR}/DoTA/DoTA_Frames" --strip-components=1
tar -xzvf "${WORK_DIR}/BDD_Frames.tar.gz" -C "${WORK_DIR}/BDD100K/BDD_Frames" --strip-components=1

echo "Data preparation complete."