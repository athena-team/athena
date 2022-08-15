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

horovod=false  # multiple-device training , set to true

#This installation script you need to do for running athena

#if [ "athena" != $(basename "$PWD") ]; then
#    echo "You should run this script in athena directory!!"
#    exit 1
#fi

# check and see if there are any system-level installations you need to do

  bash check_source.sh

# creating a virtual environment and installing the python requirements
  pip3 install virtualenv
  virtualenv venv_athena
  source venv_athena/bin/activate

# Install tensorflow backend
  pip install --upgrade pip
  pip install tensorflow==2.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install *horovod* for multiple-device training
  if $horovod;then
     bash tools/install_horovod.sh
  fi
# Install *sph2pipe*, *spm*,*sclite* for ASR Tasks
  
  bash tools/install_tools_for_asr.sh

# Other python library
  
  pip install -r requirements.txt

# Install athena package

  python setup.py bdist_wheel sdist
  python -m pip install --ignore-installed dist/athena-0.1.0*.whl
  
echo "Athena installation finished，you enjoy your Athena trip"

