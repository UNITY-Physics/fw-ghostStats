#!/bin/bash
# Choose one of the following PLATFORMS: linux-cuda, linux-cpu, mac

# Download reference dataset from figshare
dataset_id=26954638
version=1

wget --content-disposition --no-check-certificate https://figshare.com/ndownloader/articles/${dataset_id}/versions/${version}
unzip -n ${dataset_id}.zip 
rm ${dataset_id}.zip

# Build locally
# PLATFORM=linux-cpu
# docker build -t ghostseg --build-arg="PLATFORM=${PLATFORM}" --platform=linux/amd64 .

# Build flywheel
fw-beta gear config --create
# Add in your API key
fw-beta gear build

# Example of how to run interactive
fw-beta gear run -i -e bash