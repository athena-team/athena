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
stop_stage=4
horovod_cmd=""
horovod_prefix=""
#horovod_cmd="horovodrun -np 4 -H localhost:4"
#horovod_prefix="horovod_"

# please modify the following path accordingly 
# if you put dataset in the differenct path
dataset_dir=examples/asr/timit/data/TIMIT

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "Preparing data"
    mkdir -p examples/asr/timit/data || exit 1
    python examples/asr/timit/local/prepare_data.py \
        $dataset_dir examples/asr/timit/data || exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # calculate cmvn
     echo "Calculating cmvn"
    cat examples/asr/timit/data/train.csv > examples/asr/timit/data/all.csv
    tail -n +2 examples/asr/timit/data/dev.csv >> examples/asr/timit/data/all.csv
    tail -n +2 examples/asr/timit/data/test.csv >> examples/asr/timit/data/all.csv
    python  athena/cmvn_main.py \
        examples/asr/timit/configs/mtl_transformer_sp.json examples/asr/timit/data/all.csv || exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # training stage
    echo "Training"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/timit/configs/mtl_transformer_sp.json || exit 1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # decoding stage
    echo "Decoding"
    python athena/inference.py \
        examples/asr/timit/configs/mtl_transformer_sp.json || exit 1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # scoring stage
    echo "Sclite scoring"
    if ! [ -x "$(command -v ./tools/SCTK/bin/sclite)" ]; then
        echo "sclite is not installed !!"
        exit 1
    else
        log=inference.log
        score_dir=score
        mkdir -p $score_dir/score_ori $score_dir/score_map

        python examples/asr/timit/local/process_decode_result.py $log \
            examples/asr/timit/local/phones.60-48-39.map examples/asr/timit/data/vocab 48

        for file in $log.result $log.result.map $log.label $log.label.map; do
            awk '{print $0 "("NR".wav)"}' $file > $file.key
            mv $file.key $file
        done

        ./tools/SCTK/bin/sclite -i wsj -r $log.label -h $log.result -e utf-8 -o all -O $score_dir/score_ori \
            > $score_dir/score_ori/sclite.compute.log
        ./tools/SCTK/bin/sclite -i wsj -r $log.label.map -h $log.result.map -e utf-8 -o all -O $score_dir/score_map \
            > $score_dir/score_map/sclite.compute.log

        echo "original results"
        grep Err $score_dir/score_ori/inference.log.result.sys
        grep Sum/Avg $score_dir/score_ori/inference.log.result.sys
        echo "results after phoneme mapping"
        grep Err $score_dir/score_map/inference.log.result.map.sys
        grep Sum/Avg $score_dir/score_map/inference.log.result.map.sys
    fi
fi
