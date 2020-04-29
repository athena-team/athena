FROM tensorflow/tensorflow:2.1.0-gpu-py3

ENV CUDNN_VERSION=7.6.5.32-1+cuda10.1
ENV NCCL_VERSION=2.4.8-1+cuda10.1

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends  wget \
     vim \
     git \
     libcudnn7=${CUDNN_VERSION} \
     libnccl2=${NCCL_VERSION} \
     libnccl-dev=${NCCL_VERSION} 

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_WITH_TENSORFLOW=1 && \
         pip --default-timeout=1000 install --no-cache-dir git+https://github.com/horovod/horovod && \
    ldconfig

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN pip --default-timeout=1000 install sox \
        absl-py \
        yapf \
        pylint \
        flake8 \
        tqdm \
        sentencepiece \
        librosa \
        kenlm \
        pandas \
        jieba

# Install Athena
Run git clone https://github.com/didi/athena.git /athena && \
    cd athena && python setup.py bdist_wheel && \
    python -m pip install --ignore-installed dist/athena-0.1.0*.whl
