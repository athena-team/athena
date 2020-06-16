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

stage=1
stop_stage=100
horovod_cmd="horovodrun -np 4 -H localhost:4"
horovod_prefix="horovod_"
dataset_dir=examples/vc/vcc2020/data/vcc2020  #examples/vc/vcc2020/data  #/vcc2020

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "Creating csv"
    mkdir -p examples/vc/vcc2020/data
    python examples/vc/vcc2020/local/prepare_data.py \
        $dataset_dir examples/vc/vcc2020/data || exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # calculate cmvn
    echo "Computing cmvn"
    cat examples/vc/vcc2020/data/train.csv > examples/vc/vcc2020/data/all.csv
    tail -n +2 examples/vc/vcc2020/data/dev.csv >> examples/vc/vcc2020/data/all.csv
    tail -n +2 examples/vc/vcc2020/data/test.csv >> examples/vc/vcc2020/data/all.csv
    # python CUDA_VISIBLE_DEVICES='' athena/cmvn_main.py 
	python athena/cmvn_main.py \
        examples/vc/vcc2020/configs/VC.json examples/vc/vcc2020/data/all.csv || exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # pretrain stage
    echo "Pretraining"
    # $horovod_cmd python athena/${horovod_prefix}main.py \
        # examples/vc/vcc2020/configs/mpc.json || exit 1
	python athena/main.py \
		examples/vc/vcc2020/configs/mpc.json || exit 1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # finetuning stage
    echo "Fine-tuning"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/vc/vcc2020/configs/mtl_transformer_sp.json || exit 1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # prepare language model
    echo "Training language model ..."
    tail -n +2 examples/vc/vcc2020/data/train.csv |\
        cut -f 3 > examples/vc/vcc2020/data/text
    # tools/kenlm/build/bin/lmplz -o 4 < examples/vc/vcc2020/data/text \
    #     > examples/vc/aisehll/data/4gram.arpa || exit 1
    tail -n +2 examples/vc/vcc2020/data/train.csv |\
        awk '{print $3"\t"$3}' > examples/vc/vcc2020/data/train.trans.csv
    tail -n +2 examples/vc/vcc2020/data/dev.csv |\
        awk '{print $3"\t"$3}' > examples/vc/vcc2020/data/dev.trans.csv
    tail -n +2 examples/vc/vcc2020/data/test.csv |\
        awk '{print $3"\t"$3}' > examples/vc/vcc2020/data/test.trans.csv
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/vc/vcc2020/configs/rnnlm.json || exit 1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # decoding stage
    echo "Running decode ..."
    python athena/decode_main.py \
        examples/vc/vcc2020/configs/mtl_transformer_sp.json || exit 1
fi
