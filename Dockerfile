FROM ubuntu:jammy-20240427
ARG PLATFORM="linux-cpu"

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

WORKDIR /opt

# Pre reqs
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y python3 pip curl wget git

# Install pytorch
# RUN if [ "$PLATFORM" = "linux-cuda" ]; then \
#         pip3 install torch torchvision torchaudio; \
#     elif [ "$PLATFORM" = "linux-cpu" ]; then \
#         pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
#     elif [ "$PLATFORM" = "mac" ]; then \
#         pip3 install torch torchvision torchaudio; \
#     else \
#         echo "Invalid platform argument: ${PLATFORM}"; \
#     fi

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install nnUNet
RUN python3 -m pip install nnunetv2

RUN mkdir /root/ghost_data && \
    mkdir /root/ghost_data/phantom_template && \
    mkdir /root/ghost_data/nnUnet_models && \
    mkdir /root/ghost_data/nnUnet_models/nnUnet_raw && \
    mkdir /root/ghost_data/nnUnet_models/nnUnet_results && \
    mkdir /root/ghost_data/nnUnet_models/nnUnet_preprocessed

ENV nnUNet_raw=/root/ghost_data/nnUnet_models/nnUnet_raw
ENV nnUNet_results=/root/ghost_data/nnUnet_models/nnUnet_results
ENV nnUNet_preprocessed=/root/ghost_data/nnUnet_models/nnUnet_preprocessed

COPY . /usr/local/src/ghost
WORKDIR /usr/local/src/ghost
RUN python3 -m pip install .

# Import models
RUN nnUNetv2_install_pretrained_model_from_zip /usr/local/src/ghost/nnUnet_models/export237.zip
RUN nnUNetv2_install_pretrained_model_from_zip /usr/local/src/ghost/nnUnet_models/export337.zip