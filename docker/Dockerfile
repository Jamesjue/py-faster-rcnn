FROM nvidia/cuda:8.0-devel-ubuntu14.04
MAINTAINER Junjue Wang, Hongkun Leng

# install py-faster-rcnn dependency
RUN apt-get update && apt-get install -y \
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

RUN apt-get install -y --no-install-recommends libboost-all-dev

RUN wget -O /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python /tmp/get-pip.py && \
    pip install -U pip setuptools

# install py-faster-rcnn
WORKDIR /
RUN git clone --recursive  https://github.com/suanmiao/py-faster-rcnn.git

# install pip and necessary component
WORKDIR /py-faster-rcnn/caffe-fast-rcnn
RUN pip install easydict && \
    pip install -r python/requirements.txt

WORKDIR /py-faster-rcnn
RUN cd lib && \
    make -j$(nproc)

RUN cd caffe-fast-rcnn && \
    cp Makefile.config.example Makefile.config && \
    sed -i 's/BLAS := atlas/BLAS := open/' Makefile.config && \
    sed -i 's%# BLAS_INCLUDE := /path/to/your/blas%BLAS_INCLUDE := /opt/OpenBLAS/include%' Makefile.config && \
    sed -i 's%# BLAS_LIB := /path/to/your/blas%BLAS_LIB := /opt/OpenBLAS/lib%' Makefile.config && \
    sed -i 's%# WITH_PYTHON_LAYER := 1%WITH_PYTHON_LAYER := 1%' Makefile.config && \
    make -j$(nproc) && \    
    make -j$(nproc) pycaffe


#
# #RUN ./data/scripts/fetch_faster_rcnn_models.sh
#
# # use cachebust to update my commit
# ARG CACHEBUST=5
# WORKDIR /py-faster-rcnn
# RUN git pull origin master
#
# # install requirement for python detect-server
# WORKDIR /py-faster-rcnn/tools/detect-server
# RUN pip install -r requirements.txt
#
# # add model if there is one
# ADD model.caffemodel /py-faster-rcnn/models/tpod/VGG_CNN_M_1024/faster_rcnn_alt_opt/
# ADD labels.txt /py-faster-rcnn/models/tpod/VGG_CNN_M_1024/faster_rcnn_alt_opt/
# ADD faster_rcnn_test.pt /py-faster-rcnn/models/tpod/VGG_CNN_M_1024/faster_rcnn_alt_opt/
#
# # run twistd wsgi to server python application
# WORKDIR /py-faster-rcnn
# EXPOSE 8080
# CMD ["/bin/bash", "-c", "cd tools/detect-server/ && gunicorn -w 1 -b 0.0.0.0:8080 detect-web-server:app"]
