#!/bin/bash
# coding=utf-8
# Copyright (C) ATHENA AUTHORS
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# source some path
. ./tools/env.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <dataset-dir> <out_dir>"
  exit 1;
fi

dataset_dir=$1
out_dir=$2

echo "Computing cmvn........"
cat $dataset_dir/train-clean-100.csv > $out_dir/all.csv
tail -n +2 $dataset_dir/train-clean-360.csv >> $out_dir/all.csv
tail -n +2 $dataset_dir/train-other-500.csv >> $out_dir/all.csv
cp $out_dir/all.csv $out_dir/train-960.csv
tail -n +2 $dataset_dir/dev-clean.csv >> $out_dir/all.csv
tail -n +2 $dataset_dir/dev-other.csv >> $out_dir/all.csv
tail -n +2 $dataset_dir/test-clean.csv >> $out_dir/all.csv
tail -n +2 $dataset_dir/test-other.csv >> $out_dir/all.csv
CUDA_VISIBLE_DEVICES='' python athena/cmvn_main.py \
examples/asr/librispeech/configs/cmvn.json $out_dir/all.csv || exit 1

# create spm model
# spm should be installed
cat $out_dir/train-960.csv | awk -F '\t' '{print tolower($3)}' | tail -n +2 \
> $out_dir/text
spm_train --input=$out_dir/text \
--vocab_size=5000 \
--model_type=unigram \
--model_prefix=$out_dir/librispeech_unigram5000 \
--input_sentence_size=100000000


echo "all is OK！！！！！！！"
