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

#This installation script need our docker image,you can get the 
#docker image :
#and then run:
#nvidia-docker run -dit --name $docker_name --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#  -v /data/:/data jx-bd-hadoop13.zeus.lianjia.com:801/runonce/kaldi/20190916:0.0.3


# Clone athena package

  git clone https://git.lianjia.com/speech_tech/athena.git

  cd athena

if [ "athena" != $(basename "$PWD") ]; then
    echo "You should run this script in athena directory!!"
    exit 1
fi

sox=`which sox`
if [ ! -x $sox ]; then
   echo "sox is not installed !! "
   echo "Ubuntu，you can run: sudo apt-get update && sudo apt-get install sox"
   echo "Centos，you can run: sudo yum install sox"
  exit 1
fi

# check.

  bash check_source.sh

# Tool to convert sph audio format to wav format.
  
  bash tools/install_sph2pipe.sh

# Tool to compute score with sclite

  bash tools/install_sclite.sh
  bash tools/install_spm.sh

# Other python library
  
  pip install -r requirements.txt

# Install athena package

  python setup.py bdist_wheel sdist
  python -m pip install --ignore-installed dist/athena-0.1.0*.whl
  
echo "Athena installation finished"

