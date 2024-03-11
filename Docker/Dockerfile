# Start image with base image
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# Root directory used for package install
WORKDIR /


# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y python3-pip python3-dev \
    && apt-get install -y wget \
    && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Moving to /Workspace directory
WORKDIR /home/Workspace


# Installing all packages
RUN conda env create --file CRN.yaml
RUN conda activate CRN

RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pytorch-lightning==1.6.0
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchmetrics==0.4.1
RUN mim install mmcv==1.6.0
RUN mim install mmsegmentation==0.28.0
RUN mim install mmdet==2.25.2

