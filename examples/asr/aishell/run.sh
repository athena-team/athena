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

stage=1
stop_stage=100

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "making csv"
    mkdir -p examples/asr/aishell/data
    python examples/asr/aishell/local/prepare_data.py /nfs/cold_project/datasets/opensource_data/aishell/data_aishell
    cp /nfs/cold_project/datasets/opensource_data/aishell/data_aishell/{train,dev,test}.csv examples/asr/aishell/data/
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # cal cmvn
    echo "computing cmvn"
    cat examples/asr/aishell/data/train.csv > examples/asr/aishell/data/all.csv
    tail -n +2 examples/asr/aishell/data/dev.csv >> examples/asr/aishell/data/all.csv
    tail -n +2 examples/asr/aishell/data/test.csv >> examples/asr/aishell/data/all.csv
    python athena/cmvn_main.py examples/asr/aishell/mpc.json examples/asr/aishell/data/all.csv
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # pretrain stage
    echo "Pretraining"
    python athena/main.py examples/asr/aishell/mpc.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # finetuning stage
    echo "Fine-tuning"
    python athena/main.py examples/asr/aishell/mtl_transformer_sp.json
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # prepare language model
    echo "training language model ..."
    tail -n +2 examples/asr/aishell/data/train.csv | cut -f 3 > examples/asr/aishell/data/text
    # tools/kenlm/build/bin/lmplz -o 4 < examples/asr/aishell/data/text > examples/asr/aisehll/data/4gram.arpa
    tail -n +2 examples/asr/aishell/data/train.csv | awk '{print $3"\t"$3}' > examples/asr/aishell/data/train.trans.csv
    tail -n +2 examples/asr/aishell/data/dev.csv | awk '{print $3"\t"$3}' > examples/asr/aishell/data/dev.trans.csv
    tail -n +2 examples/asr/aishell/data/test.csv | awk '{print $3"\t"$3}' > examples/asr/aishell/data/test.trans.csv
    python athena/main.py examples/asr/aishell/rnnlm.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # decoding stage
    echo "running decode ..."
    python athena/decode_main.py examples/asr/aishell/mtl_transformer_sp.json
fi
