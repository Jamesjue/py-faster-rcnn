#!/usr/bin/env bash
# install py-faster-rcnn dependency

sudo apt-get update && sudo apt-get install -y \
    build-essential \
    git \
    libopencv-dev \
    python-opencv \
    python-dev \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    libhdf5-serial-dev \
    protobuf-compiler \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev \
    libopenblas-base \
    libopenblas-dev \
    wget

apt-get install -y --no-install-recommends libboost-all-dev

# install pip and necessary component
pip install easydict

cd lib
make -j$(nproc)

# should be removed in Docker file
cd ..

cd caffe-fast-rcnn 
cp Makefile.config.example Makefile.config  
sed -i 's/BLAS := atlas/BLAS := open/' Makefile.config 
sed -i 's%# BLAS_INCLUDE := /path/to/your/blas%BLAS_INCLUDE := /opt/OpenBLAS/include%' Makefile.config 
sed -i 's%# BLAS_LIB := /path/to/your/blas%BLAS_LIB := /opt/OpenBLAS/lib%' Makefile.config 
sed -i 's%# WITH_PYTHON_LAYER := 1%WITH_PYTHON_LAYER := 1%' Makefile.config && \
make -j$(nproc) 
make -j$(nproc) pycaffe

#RUN ./data/scripts/fetch_faster_rcnn_models.sh



