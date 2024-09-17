#!/bin/bash
# Choose one of the following PLATFORMS: linux-cuda, linux-cpu, mac

PLATFORM=mac
docker build -t ghost --build-arg="PLATFORM=${PLATFORM}" .