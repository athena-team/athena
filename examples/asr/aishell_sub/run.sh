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

if [ "athena" != $(basename "$PWD") ]; then
    echo "You should run this script in athena directory!!"
    exit 1
fi

source tools/env.sh

stage=0
stop_stage=100
dataset_dir=examples/asr/aishell/data/data_aishell
workplace_dir=examples/asr/aishell_sub

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "Creating csv"
    mkdir -p ${workplace_dir}/data
    python ${workplace_dir}/local/prepare_data.py \
        $dataset_dir ${workplace_dir}/data || exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # calculate cmvn
    echo "Computing cmvn"
    cat ${workplace_dir}/data/train.csv > ${workplace_dir}/data/all.csv
    tail -n +2 ${workplace_dir}/data/dev.csv >> ${workplace_dir}/data/all.csv
    tail -n +2 ${workplace_dir}/data/test.csv >> ${workplace_dir}/data/all.csv
    CUDA_VISIBLE_DEVICES='' python athena/cmvn_main.py \
        ${workplace_dir}/configs/mpc.json ${workplace_dir}/data/all.csv || exit 1
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # finetuning stage
    echo "Fine-tuning"
    python athena/main.py \
        ${workplace_dir}/configs/mtl_transformer_sp.json || exit 1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # prepare language model
    echo "Training language model ..."
    tail -n +2 ${workplace_dir}/data/train.csv |\
        cut -f 3 > ${workplace_dir}/data/text
    tail -n +2 ${workplace_dir}/data/train.csv |\
        awk '{print $3"\t"$3}' > ${workplace_dir}/data/train.trans.csv
    tail -n +2 ${workplace_dir}/data/dev.csv |\
        awk '{print $3"\t"$3}' > ${workplace_dir}/data/dev.trans.csv
    tail -n +2 ${workplace_dir}/data/test.csv |\
        awk '{print $3"\t"$3}' > ${workplace_dir}/data/test.trans.csv
    python athena/main.py \
        ${workplace_dir}/configs/rnnlm.json || exit 1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # decoding stage
    echo "Running decode ..."
    python athena/inference.py \
        ${workplace_dir}/configs/mtl_transformer_sp.json || exit 1
fi
