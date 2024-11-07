FROM nialljb/ghost-base:0.0.1 AS base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone handling
ENV CONTAINER_TIMEZONE=UTC
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

WORKDIR /opt

# Install system dependencies
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y python3 pip curl wget git cmake libpng-dev

# Configure Flywheel
ENV FLYWHEEL="/flywheel/v0"
WORKDIR $FLYWHEEL

# Instal Flywheel depedencies
COPY ./ $FLYWHEEL/
RUN pip3 install flywheel-gear-toolkit && \
    pip3 install --upgrade flywheel-sdk

# Install ghost
RUN pip3 install .

# Fix directory names
RUN mkdir /root/ghost_data/nnUNet/nnUNet_results && \
    ln -s /root/ghost_data/nnUnet_models/nnUnet_results/Dataset227_UNITY /root/ghost_data/nnUNet/nnUNet_results/Dataset237_UNITY && \
    ln -s /root/ghost_data/nnUnet_models/nnUnet_results/Dataset237_UNITY /root/ghost_data/nnUNet/nnUNet_results/Dataset337_UNITY && \
    ln -s /root/ghost_data/nnUnet_models/nnUnet_results/Dataset247_UNITY /root/ghost_data/nnUNet/nnUNet_results/Dataset437_UNITY

ENTRYPOINT ["python3","/flywheel/v0/run.py"] 