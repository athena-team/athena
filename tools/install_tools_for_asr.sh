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

sox=`which sox`
if [ ! -x $sox ]; then
   echo "sox is not installed !! "
   echo "Ubuntu，you can run: sudo apt-get update && sudo apt-get install sox"
   echo "Centos，you can run: sudo yum install sox"
  exit 1
fi

# Tool to convert sph audio format to wav format.
  
  bash tools/install_sph2pipe.sh &>install_sph2pipe.log.txt

# Tool to compute score with sclite
  bash tools/install_sclite.sh &>install_sclite.log.txt

# Tool to deal text

  bash tools/install_spm.sh &>install_spm.log.txt

# install flac
  wget  https://downloads.xiph.org/releases/flac/flac-1.3.3.tar.xz
  xz -d flac-1.3.3.tar.xz
  tar -xvf flac-1.3.3.tar
  mkdir -p /usr/local/flac
  cd flac-1.3.3
  ./configure --prefix=/usr/local/flac
  make && make install
  echo "export PATH=/usr/local/flac/bin:$PATH">>~/.bashrc
  source ~/.bashrc
