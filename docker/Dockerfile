FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV NCCL_VERSION=2.8.3-1+cuda10.1

# install python environments
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends python3 \
     python3-venv \
     python3-pip \
<<<<<<< HEAD
     python-setuptools \
=======
     python-setuptools \ 
>>>>>>> 7e867948a35a06b2e5db57c93f10f153b6967f43
     python3.6-dev \
     python3-dev \
     python3-setuptools

# wget git sox cmake NCCL2

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends  wget \
     vim \
     git \
     sox \
     cmake \
     libnccl2=${NCCL_VERSION} \
     libnccl-dev=${NCCL_VERSION}

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz && \
    tar zxf openmpi-4.0.5.tar.gz && \
    cd openmpi-4.0.5 && \
    ./configure --prefix=/usr/local && \
    make all install && \
    cd - && \
    echo "export PATH=/usr/local/lib:$PATH">>~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH">>~/.bashrc && \
    rm -rf /tmp/openmpi
#
RUN pip3 install wheel
#
# Install Horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL pip3 --default-timeout=1000 install horovod

#Install tensorflow backend
RUN python3 -m pip install --default-timeout=1000 --upgrade --force-reinstall pip && \
        pip3 install --default-timeout=1000 tensorflow==2.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# install other python library
RUN pip3 --default-timeout=1000 install sox \
        librosa==0.8.0 \
	absl-py \
        yapf \
        pylint \
        flake8 \
        tqdm \
        sentencepiece \
        jieba \
        pandas==0.25.1 \
        kaldiio \
        rarfile \
        inflect \
        pyworld \
        Pillow \
        https://github.com/kpu/kenlm/archive/master.zip \
	tensorflow-text==2.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# clone *athena* package and install athena package
RUN git clone https://github.com/LianjiaTech/athena.git && \
    cd athena && python setup.py bdist_wheel && \
    python -m pip install --ignore-installed dist/athena-2.0*.whl
<<<<<<< HEAD

=======
    
>>>>>>> 7e867948a35a06b2e5db57c93f10f153b6967f43
# clone *athena* package and install athena package
RUN sh tools/install_sph2pipe.sh && \
    apt-get install -y build-essential libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev && \
    sh tools/install_sclite.sh && \
    sh tools/install_spm.sh
