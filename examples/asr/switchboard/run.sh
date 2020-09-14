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

if [ "athena" != $(basename "$PWD") ]; then
    echo "You should run this script in athena directory!!"
    exit 1
fi

source tools/env.sh

stage=0
stop_stage=5

horovod_cmd=""
horovod_prefix=""
# uncomment the following commands if you want to use horovod
# horovod_cmd="horovodrun -np 4 -H localhost:4"
# horovod_prefix="horovod_"

# please modify the following pathes accordingly 
# if you put dataset or sctk in differenct pathes
dataset_dir=examples/asr/switchboard/data
sctk_path=tools/sctk

bpe_prefix=examples/asr/switchboard/data/switchboard_bpe2000
score_dir=examples/asr/switchboard/score
decode_log=inference.log
mkdir -p ${score_dir}


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    # we reuse the prepare_data.py script in "switchboard_fisher" here
    echo "Preparing data"
    mkdir -p examples/asr/switchboard/data || exit 1
    for set in switchboard fisher hub500; do
        python examples/asr/switchboard_fisher/prepare_data.py \
            $dataset_dir examples/asr/switchboard/data ${set} || exit 1
    done

    # calculate cmvn
    cp examples/asr/switchboard/data/switchboard.csv examples/asr/switchboard/data/all.csv
    tail -n +2 examples/asr/switchboard/data/hub500.csv >> examples/asr/switchboard/data/all.csv
    CUDA_VISIBLE_DEVICES='' python athena/cmvn_main.py \
        examples/asr/switchboard/configs/mpc.json examples/asr/switchboard/data/all.csv || exit 1
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # create spm model
    # spm should be installed
    echo "Create spm model"
    cat examples/asr/switchboard/data/switchboard.csv | awk -F '\t' '{print tolower($3)}' | tail -n +2 \
        > examples/asr/switchboard/data/text

    # character_coverage should be set to 1.0 in order to cover all characters
    spm_train --input=examples/asr/switchboard/data/text \
            --vocab_size=2000 \
            --character_coverage=1.0 \
            --model_type=bpe \
            --model_prefix=${bpe_prefix} \
            --input_sentence_size=100000000 \
            --bos_id=-1 \
            --eos_id=-1 \
            --unk_id=0

    # users can encode transcripts in csvs into bpe pieces and modify following scripts accordingly
    # to avoid conficts between horovod and sentencepiece
    # for set in switchboard fisher hub500; do
    #     cat examples/asr/switchboard/data/${set}.csv | awk -F '\t' '{print tolower($3)}' | tail -n +2 \
    #             > examples/asr/switchboard/data/text
    #     spm_encode --model=${bpe_prefix}.model --output_format=piece < examples/asr/switchboard/data/text > \
    #             examples/asr/switchboard/data/text.encoded
    #     sed -i '1 i\transcript' examples/asr/switchboard/data/text.encoded
    #     awk -F "\t" '{getline trans < "examples/asr/switchboard/data/text.encoded"; print $1"\t"$2"\t"trans"\t"$4}' < \
    #             examples/asr/switchboard/data/${set}.csv > examples/asr/switchboard/data/${set}.bpe2000.csv
    #     mv examples/asr/switchboard/data/${set}.csv examples/asr/switchboard/data/${set}.csv.bk
    #     mv examples/asr/switchboard/data/${set}.bpe2000.csv examples/asr/switchboard/data/${set}.csv
    # done
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # training stage
    echo "Training"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/switchboard/configs/mtl_transformer_sp.json || exit 1
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # prepare language model
    echo "Training language model ..."
    cat examples/asr/switchboard/data/switchboard.csv | \
        awk -F '\t' '{print $3"\t"$3}' > examples/asr/switchboard/data/train.trans.csv
    tail -n +2 examples/asr/switchboard/data/fisher.csv | \
        awk -F '\t' '{print $3"\t"$3}' >> examples/asr/switchboard/data/train.trans.csv
    cat examples/asr/switchboard/data/hub500.csv | \
        awk -F '\t' '{print $3"\t"$3}' > examples/asr/switchboard/data/dev.trans.csv
    python athena/main.py \
        examples/asr/switchboard/configs/rnnlm.json || exit 1
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # decoding stage
    echo "Running decode ..."
    python athena/inference.py \
        examples/asr/switchboard/configs/mtl_transformer_sp.json || exit 1
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    # scoring stage, scoring only the swbd part in hub500
    echo "Running sclite scoring ..."
    # copy 'stm' and 'glm' of hub500 to score_dir for scoring
    cat ${dataset_dir}/LDC2002T43/reference/hub5e00.english.000405.stm | \
        sed -e 's:((:(:' -e 's:<B_ASIDE>::g' -e 's:<E_ASIDE>::g' > ${score_dir}/stm
    cp ${dataset_dir}/LDC2002T43/reference/en20000405_hub5.glm ${score_dir}/glm

    # sclite should be installed
    ./examples/asr/switchboard/local/score_sclite_hub.sh ${decode_log} ${bpe_prefix}.model ${score_dir} ${sctk_path}
fi
