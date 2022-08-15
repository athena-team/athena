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
gpu_num=8
train_conf=examples/asr/misp/conf/mtl_conformer_sp_fbank80_batch_bins_av.json
deocode_conf=examples/asr/misp/conf/mtl_conformer_sp_fbank80_batch_bins_av_decode.json


# seletion options
rnnlm=true  # rnn language model training is provided ,set to false,if use ngram language model


# source some path
. ./tools/env.sh

if [ "athena" != $(basename "$PWD") ]; then
    echo "You should run this script in athena directory!!"
    exit 1
fi

#  ASR training stage
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # training command
  if [ $gpu_number -gt 1 ]; then
    echo "Multi-gpu training"
    echo "horovodrun -np $gpu_num -H localhost:$gpu_num python athena/horovod_main.py $train_conf"
    horovodrun -np $gpu_num -H localhost:$gpu_num python athena/horovod_main.py $train_conf
  else:
    echo "Single-gpu training"
    echo "athena/main.py $train_conf"
    python athena/main.py $train_conf
  fi
fi

# language model trianing
if $rnnlm;then
   # training rnnlm
   if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "training rnnlm"
		bash examples/asr/misp/local/misp_train_rnnlm.sh
   fi
else
   # training ngram lm
   if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
        echo "training ngram lm"
		bash examples/asr/misp/local/misp_train_lm.sh
   fi
fi

# decode

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Decoding"
    python athena/inference.py \
        $decode_conf || exit 1
fi

# score-computing stage

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
   echo "computing score with sclite ..."
   bash examples/asr/hkust/local/run_score.sh inference.log score_hkust examples/asr/hkust/data/vocab
fi

echo "$0: Finished MISP training examples"
