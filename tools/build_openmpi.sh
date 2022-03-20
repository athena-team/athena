#!/bin/bash
# coding=utf-8
# Copyright (C) ATHENA AUTHORS, JianweiSun
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
work_dir=`pwd`
function gen_openmpi_env_file()
{
  echo "export LD_LIBRARY_PATH=$work_dir/openmpi-4.0.2/lib:\$LD_LIBRARY_PATH">openmpi.env
  echo "export PATH=$work_dir/openmpi-4.0.2/bin:\$PATH">>openmpi.env
}

function build_openmpi()
{
if [ ! -f openmpi-4.0.2.tar.gz ];then
  echo wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz
  wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz
fi
if [ ! -f openmpi-4.0.2 ];then
  tar zxvf openmpi-4.0.2.tar.gz
  cd openmpi-4.0.2
  ./configure --prefix=$work_dir/openmpi-4.0.2/bin
  make clean
  make -j16
  make install
  cd -
fi
}
function main()
{
build_openmpi
gen_openmpi_env_file
}
main
