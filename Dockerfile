FROM nialljb/fw-ghost-base as base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone handling
ENV CONTAINER_TIMEZONE=UTC
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

WORKDIR /opt

# Install system dependencies
RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y python3 pip curl wget git cmake  libpng-dev


# Upgrade pip and install Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    nnunetv2

# Set nnUNet environment variables
ENV nnUNet_raw=/root/ghost_data/nnUnet_models/nnUnet_raw
ENV nnUNet_results=/root/ghost_data/nnUnet_models/nnUnet_results
ENV nnUNet_preprocessed=/root/ghost_data/nnUnet_models/nnUnet_preprocessed

# Install project
COPY . /usr/local/src/ghost
WORKDIR /usr/local/src/ghost
RUN python3 -m pip install .

# Configure Flywheel
ENV FLYWHEEL="/flywheel/v0"
WORKDIR $FLYWHEEL

# Instal Flywheel depedencies
COPY ./ $FLYWHEEL/
RUN pip3 install flywheel-gear-toolkit && \
    pip3 install --upgrade flywheel-sdk

ENTRYPOINT ["python3","/flywheel/v0/run.py"] 