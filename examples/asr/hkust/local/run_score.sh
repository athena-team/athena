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

if [ $# -ne 3 ]; then
  echo "Usage: $0 <log-dir> <score-dir> <vocab-file>"
  exit 1;
fi

log=$1
score_dir=$2
vocab=$3

if  [ ! -x "$(command -v ./tools/SCTK/bin/sclite)" ]; then
        echo "sclite is not installed !! "
		echo "Checkout tools/install_sclite.sh, run tools/install_sclite.sh to install"
        exit 1
    else
        mkdir -p $score_dir
        python athena/tools/process_decode_result.py $log $vocab

        for file in $log.result $log.label; do
            awk '{print $0 "("NR".wav)"}' $file > $file.key
            mv $file.key $file
        done

        ./tools/SCTK/bin/sclite -i wsj -r $log.label -h $log.result -e utf-8 -o all -O $score_dir \
            > $score_dir/sclite.compute.log
        grep Err $score_dir/inference.log.result.sys
        grep Sum/Avg $score_dir/inference.log.result.sys
fi
