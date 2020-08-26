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
horovod_cmd="horovodrun -np 4 -H localhost:4"
horovod_prefix="horovod_"
dataset_dir=examples/asr/librispeech/data

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "Creating csv"
    mkdir -p examples/asr/librispeech/data
    python examples/asr/librispeech/local/prepare_data.py \
        $dataset_dir || exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # calculate cmvn
    echo "Computing cmvn"
    cat $dataset_dir/train-clean-100.csv > examples/asr/librispeech/data/all.csv
    tail -n +2 $dataset_dir/train-clean-360.csv >> examples/asr/librispeech/data/all.csv
    tail -n +2 $dataset_dir/train-other-500.csv >> examples/asr/librispeech/data/all.csv
    cp examples/asr/librispeech/data/all.csv examples/asr/librispeech/data/train-960.csv
    tail -n +2 $dataset_dir/dev-clean.csv >> examples/asr/librispeech/data/all.csv
    tail -n +2 $dataset_dir/dev-other.csv >> examples/asr/librispeech/data/all.csv
    tail -n +2 $dataset_dir/test-clean.csv >> examples/asr/librispeech/data/all.csv
    tail -n +2 $dataset_dir/test-other.csv >> examples/asr/librispeech/data/all.csv
    CUDA_VISIBLE_DEVICES='' python athena/cmvn_main.py \
        examples/asr/librispeech/configs/mpc.json examples/asr/librispeech/data/all.csv || exit 1

    # create spm model
    # spm should be installed
    cat examples/asr/librispeech/data/train-960.csv | awk -F '\t' '{print tolower($3)}' | tail -n +2 \
        > examples/asr/librispeech/data/text
    spm_train --input=examples/asr/librispeech/data/text \
        --vocab_size=5000 \
        --model_type=unigram \
        --model_prefix=examples/asr/librispeech/data/librispeech_unigram5000 \
        --input_sentence_size=100000000
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # pretrain stage
    echo "Pretraining"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/librispeech/configs/mpc.json || exit 1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # finetuning stage
    echo "Fine-tuning"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/librispeech/configs/mtl_transformer_sp.json || exit 1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # prepare language model
    echo "Training language model ..."
    # tools/kenlm/build/bin/lmplz -o 4 < examples/asr/librispeech/data/text \
    #     > examples/asr/aisehll/data/4gram.arpa || exit 1
    cat examples/asr/librispeech/data/train-960.csv |\
        awk -F '\t' '{print $3"\t"$3}' > examples/asr/librispeech/data/train.trans.csv
    cat examples/asr/librispeech/data/dev-clean.csv |\
        awk -F '\t' '{print $3"\t"$3}' > examples/asr/librispeech/data/dev.trans.csv
    cat examples/asr/librispeech/data/test-clean.csv |\
        awk -F '\t' '{print $3"\t"$3}' > examples/asr/librispeech/data/test.trans.csv
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/librispeech/configs/rnnlm.json || exit 1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # decoding stage
    echo "Running decode ..."
    python athena/inference.py \
        examples/asr/librispeech/configs/mtl_transformer_sp.json || exit 1
fi
