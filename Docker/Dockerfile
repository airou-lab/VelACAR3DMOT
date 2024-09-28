# Start image with base image
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# Root directory used for package install
WORKDIR /


# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y python3-pip python3-dev \
    && apt-get install -y wget \
    && apt-get install -y jq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# getting .yaml configuration file from /Docker folder to container /tmp folder
COPY velacar.yaml /tmp/velacar.yaml

# Installing all packages and creating conda environment
RUN conda env create --file /tmp/velacar.yaml
#RUN conda init
#RUN conda activate velacar
RUN echo "source activate velacar" > ~/.bashrc
ENV PATH /opt/conda/envs/velacar/bin:$PATH

# Detection libs
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pytorch-lightning==1.6.0
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchmetrics==0.4.1
RUN mim install mmcv==1.6.0
RUN mim install mmsegmentation==0.28.0
RUN mim install mmdet==2.25.2

# Tracking libs
RUN pip install filterpy==1.4.5
RUN pip install motmetrics==1.1.3
RUN pip install nuscenes-devkit==1.1.11

# Moving to /ws (workspace) directory
WORKDIR /home/ws
