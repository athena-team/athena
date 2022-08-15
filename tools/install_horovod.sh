#!/bin/bash
# coding=utf-8
# Copyright (C) ATHENA AUTHORS
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

## install NCCL2
# you should choose the version corresponding to your CUDA and ubuntu,eg:cuda 10.1,ubuntu1804, if the following network installer is failed,you can use local installer（sudo dpkg -i nccl-repo-ubuntu1804-2.8.3-ga-cuda10.1_1-1_amd64.deb）,the website:https://developer.nvidia.com/nccl/nccl-download

sudo apt update
sudo apt install libnccl2=2.8.3-1+cuda10.1 libnccl-dev=2.8.3-1+cuda10.1


## install OpenMPI

if [ ! -f openmpi-4.0.5.tar.gz ];then

    echo 'wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz'
    wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz
fi

if [ ! -f openmpi-4.0.5 ];then
    tar zxvf openmpi-4.0.5.tar.gz
    cd openmpi-4.0.5
    ./configure --prefix=/usr/local
    make all install
    cd -
    echo "export PATH=/usr/local/lib:$PATH">>~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH">>~/.bashrc
fi


## install Cmake

sudo apt install cmake

## install horovod

HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 pip install horovod==0.23.0

horovodrun=`which horovodrun`
if [ ! -x $horovodrun ]; then
   echo "horovod maybe not installed successfully"
   echo "Please check above steps！！"
  exit 1
else
   echo "Horovod installation finished"
fi


