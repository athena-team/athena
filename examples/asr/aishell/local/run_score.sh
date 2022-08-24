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

# score-computing stage,computing score with sclite
# source some path
. ./tools/env.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <result-prefix> <vocab-file>"
  exit 1;
fi

result_prefix=$1
vocab=$2
score_dir=`dirname ${result_prefix}`/score

mkdir -p $score_dir

if  [ ! -x "$(command -v ./tools/SCTK/bin/sclite)" ]; then
        echo "sclite is not installed !! "
		echo "Checkout tools/install_sclite.sh, run tools/install_sclite.sh to install"
        exit 1
    else
        ./tools/SCTK/bin/sclite -f 1 -i wsj -r ${result_prefix}.label -h ${result_prefix}.result \
            -e utf-8 -o all -O $score_dir -f 1
fi

