FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6"
ENV MAX_JOBS=1

# 1) System packages + Python 3.8 (native on Ubuntu 20.04)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget git unzip cmake build-essential \
        python3-dev python3-pip \
        libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
        openmpi-bin openmpi-common libopenmpi-dev && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2) PyTorch 1.10.1 + CUDA 11.3 prebuilt wheels
RUN python -m pip install --no-cache-dir \
    torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 3) Prebuilt mmcv-full wheel + Python dependencies
RUN python -m pip install --no-cache-dir \
    mmcv-full==1.4.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html && \
    python -m pip install --no-cache-dir \
    mmdet==2.20.0 Pillow==8.4.0 tqdm torchpack nuscenes-devkit mpi4py==3.0.3 numba==0.57.1
