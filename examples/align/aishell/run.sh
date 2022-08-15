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

set -e

# train options
stage=4 # default training is done
stop_stage=100

gpu_nums=1

# seletion options
pretrain=false    # pretrain options, we provide Masked Predictive Coding (MPC) pretraining, default false
offline=true   # use offline feature, also can use feature file in kaldi format.
align_config=examples/align/aishell/configs/mtl_transformer_sp_fbank80.align.json

# source some path
. ./tools/env.sh

# horovod options
horovod_cmd=
horovod_prefix=
if [ ${gpu_nums} -gt 1 ]; then
  horovod_cmd="horovodrun -np ${gpu_nums} -H localhost:${gpu_nums}"
  horovod_prefix="horovod_"
fi

if [ "athena" != $(basename "$PWD") ]; then
    echo "You should run this script in athena directory!!"
    exit 1
fi

# data options
dataset_dir=examples/asr/aishell/data/data_aishell


if [ ! -d "$dataset_dir" ]; then
  echo "\no such directory $dataset_dir"
  echo "downloading data from www.openslr.org..... "
  bash examples/asr/aishell/local/aishell_download_and_untar.sh examples/asr/aishell/data \
                                                            https://openslr.magicdatatech.com/resources/33 \
														   data_aishell
fi

# prepare aishell data
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then

    echo "Preparing data and Creating csv"
    bash examples/asr/aishell/local/aishell_data_prep.sh $dataset_dir examples/asr/aishell/data
fi

# calculate cmvn
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "extract cmvn"
    cat examples/asr/aishell/data/train.csv > examples/asr/aishell/data/all.csv
    tail -n +2 examples/asr/aishell/data/dev.csv >> examples/asr/aishell/data/all.csv
    tail -n +2 examples/asr/aishell/data/test.csv >> examples/asr/aishell/data/all.csv
    CUDA_VISIBLE_DEVICES='' python athena/cmvn_main.py \
        examples/asr/aishell/configs/cmvn.json examples/asr/aishell/data/all.csv || exit 1
fi

## pretrain stage
if $pretrain;then
   if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Pretraining with mpc"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/aishell/configs/mpc.json || exit 1
   fi
fi

# Multi-task training stage
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Multi-task training"
    if ! $offline; then
      $horovod_cmd python athena/${horovod_prefix}main.py \
          examples/asr/aishell/configs/mtl_transformer_sp_fbank80.json || exit 1
    else
      if [ ! -f examples/asr/aishell/data/train/.done ]; then
          python athena/tools/storage_features_offline.py examples/asr/aishell/configs/mtl_transformer_sp_offline.json || exit 1
          touch examples/asr/aishell/data/train/.done
      fi
      $horovod_cmd python athena/${horovod_prefix}main.py \
          examples/asr/aishell/configs/mtl_transformer_sp_offline.json || exit 1
    fi
fi

# CTC Alignment
   # attention_rescoring decoding, transformer_lm rescoring default.
   if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Running CTC alignment"
      mkdir -p examples/align/aishell/result
      python athena/alignment.py $align_config || exit 1
   fi
fi

echo "$0: Finished AISHELL alignment examples"
