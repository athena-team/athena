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

# source some path
. ./tools/env.sh

# rnn language model

if [ $# -ne 2 ]; then
  echo "Usage: $0 <dataset-dir> <out-dir>"
  exit 1;
fi

dataset_dir=$1
out_dir=$2

mkdir -p $out_dir || exit 1

if [ ! -d "$out_dir/train" ];then
	   
	rm -rf $out_dir/train || exit 1
fi
	
if [ ! -d "$out_dir/dev" ];then
	   
	rm -rf $out_dir/dev || exit 1
fi

python examples/asr/hkust/local/prepare_data.py \
        $dataset_dir $out_dir || exit 1
