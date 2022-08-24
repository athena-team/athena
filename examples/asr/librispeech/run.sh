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
stage=0
stop_stage=100

# horovod options

horovod_cmd="horovodrun -np 1 -H localhost:1"
horovod_prefix="horovod_"

# seletion options 
pretrain=false    # pretrain options, we provide Masked Predictive Coding (MPC) pretraining, default false 

# source some path
. ./tools/env.sh

if [ "athena" != $(basename "$PWD") ]; then
    echo "You should run this script in athena directory!!"
    exit 1
fi

# download data

dataset_dir=examples/asr/librispeech/data/data_librispeech
data_url=www.openslr.org/resources/12

if [ ! -d "$dataset_dir" ]; then
  echo "no such directory $dataset_dir"
  echo "downloading data from www.openslr.org..... "
  mkdir -p $dataset_dir
  for part in dev-clean dev-other test-clean test-other train-clean-100 train-clean-360 train-other-500; do
    bash examples/asr/librispeech/local/librispeech_download_and_untar.sh $dataset_dir $data_url $part
  done
fi

# prepare librispeech data

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    
    echo "Preparing data and Creating csv"
    bash examples/asr/librispeech/local/librispeech_data_prep.sh $dataset_dir
fi

# calculate cmvn

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # calculate cmvn and create spm model
    bash examples/asr/librispeech/local/librispeech_cal_cmvn_spm.sh $dataset_dir examples/asr/librispeech/data
fi

# storage features offline

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && $offline; then
    echo "storage features offline"
    python athena/tools/storage_features_offline.py examples/asr/librispeech/configs/storage_features_offline.json
fi


if $pretrain;then
   if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Pretraining with mpc"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/librispeech/configs/mpc.json || exit 1
   fi
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # finetuning stage
    echo "Multi-task training"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/librispeech/configs/mtl_transformer_sp.json || exit 1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # prepare language model
    echo "Training language model ..."
	bash examples/asr/librispeech/local/librispeech_train_rnnlm.sh
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    # decoding stage
    echo "Running decode ..."
    python athena/inference.py \
        examples/asr/librispeech/configs/mtl_transformer_sp.json || exit 1
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
   echo "computing score with sclite ..."
   bash examples/asr/librispeech/local/run_score.sh inference.log score_librispeech examples/asr/librispeech/data/vocab
fi


echo "$0: Finished Librispeech training examples"
