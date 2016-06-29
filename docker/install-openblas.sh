#!/usr/bin/env bash
set -e

install_openblas() {
    # Get and build OpenBlas (Torch is much better with a decent Blas)
    cd /tmp/
    rm -rf OpenBLAS
    git clone https://github.com/xianyi/OpenBLAS.git
    cd OpenBLAS
    if [ $(getconf _NPROCESSORS_ONLN) == 1 ]; then
        make NO_AFFINITY=1 USE_OPENMP=0 USE_THREAD=0
    else
        echo "multiple cpu found. utilizing multiple threads"
        make NO_AFFINITY=1 USE_OPENMP=1
    fi
    RET=$?;
    if [ $RET -ne 0 ]; then
        echo "Error. OpenBLAS could not be compiled";
        exit $RET;
    fi
    sudo make install
    RET=$?;
    if [ $RET -ne 0 ]; then
        echo "Error. OpenBLAS could not be installed";
        exit $RET;
    fi

    # configure system wide variables
    sudo touch /etc/profile.d/openblas.sh
    echo "export PATH=/opt/OpenBLAS/bin:$PATH" | sudo tee /etc/profile.d/openblas.sh
    echo "export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:$LD_LIBRARY_PATH" | sudo tee /etc/profile.d/openblas.sh
    
    # clean up
    #Minimize image size (gfortran is needed at runtime)
#    apt-get remove -y --purge git-core build-essential
    apt-get autoremove -y
    apt-get clean -y
}

install_openblas || true
