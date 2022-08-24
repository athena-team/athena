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

gpu_nums=1

# seletion options
pretrain=false    # pretrain options, we provide Masked Predictive Coding (MPC) pretraining, default false
train_lm=true
transformer_lm=true  # rnn language model training is provided ,set to false,if use ngram language model
use_wfst=false  # decode options
offline=true   # use offline feature, also can use feature file in kaldi format.

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

if $train_lm; then
  decode_config=examples/asr/aishell/configs/mtl_transformer_sp_fbank80_decode_transformer_lm.json
else
  decode_config=examples/asr/aishell/configs/mtl_transformer_sp_fbank80_decode.json
fi

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

# prepare language model
if $train_lm; then
  if $transformer_lm; then
     # training transformer_lm
     if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
          echo "training transformer_lm"
      bash examples/asr/aishell/local/aishell_train_transformer_lm.sh examples/asr/aishell/configs/transformer_lm.json
     fi
  else
     # training ngram lm
     if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
          echo "training ngram lm"
      bash examples/asr/aishell/local/aishell_train_ngram_lm.sh examples/asr/aishell/data/5gram.arpa
     fi
  fi
fi

# decode
if $use_wfst;then
   # wfst decoding
   if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Running decode with WFST..."
      echo "For now we will simply download LG.fst and words.txt from athena-decoder project"
      echo "Feel free to checkout graph creation manual at https://github.com/athena-team/athena-decoder#build-graph"
      git clone https://github.com/athena-team/athena-decoder
      cp athena-decoder/examples/aishell/graph/LG.fst examples/asr/aishell/data/
      cp athena-decoder/examples/aishell/graph/words.txt examples/asr/aishell/data/
      python athena/inference.py \
          examples/asr/aishell/configs/mtl_transformer_sp_wfst.json || exit 1
   fi
else
   # attention_rescoring decoding, transformer_lm rescoring default.
   # attention_rescoring, ctc_prefix_beam_search and attention_decoder decoding are optional.
   if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Running decode with attention rescoring..."
      mkdir -p examples/asr/aishell/result
      python athena/inference.py $decode_config || exit 1
   fi
fi

# score-computing stage

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
   echo "computing score with sclite ..."
   result_path_without_ext=$(grep predict_path $decode_config | awk -F "[\"\"]" '{print $4}' | awk -F '.' '{$NF="";print $0}')
   bash examples/asr/aishell/local/run_score.sh $result_path_without_ext examples/asr/aishell/data/vocab
fi


echo "$0: Finished AISHELL training examples"
