# coding=utf-8
# Copyright (C) ATHENA AUTHORS: Shuaijiang Zhao, Xiaoning Lei
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

horovod_cmd="horovodrun -np 8 -H localhost:8"
horovod_prefix="horovod_"
dataset_dir=your/path/gigaspeech/

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "Creating csv"
    mkdir -p examples/asr/gigaspeech/data
    python examples/asr/gigaspeech/local/prepare_data.py \
        $dataset_dir examples/asr/gigaspeech/data || exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # calculate cmvn
    echo "Computing cmvn"
    cat examples/asr/gigaspeech/data/train.csv > examples/asr/gigaspeech/data/all.csv
    tail -n +2 examples/asr/gigaspeech/data/dev.csv >> examples/asr/gigaspeech/data/all.csv
    tail -n +2 examples/asr/gigaspeech/data/test.csv >> examples/asr/gigaspeech/data/all.csv
    CUDA_VISIBLE_DEVICES='' python athena/cmvn_main.py \
        examples/asr/gigaspeech/configs/mpc.json examples/asr/gigaspeech/data/all.csv || exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # pretrain stage
    echo "Pretraining"
    echo $horovod_cmd python athena/${horovod_prefix}main.py examples/asr/gigaspeech/configs/mpc.json
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/gigaspeech/configs/mpc.json || exit 1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # finetuning stage
    echo "Fine-tuning"
    echo $horovod_cmd python athena/${horovod_prefix}main.py examples/asr/gigaspeech/configs/mtl_transformer_sp.json
    echo $horovod_cmd python athena/${horovod_prefix}main.py examples/asr/gigaspeech/configs/mtl_transformer_sp.json | sh

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    # prepare language model
    echo "Training language model ..."
    tail -n +2 examples/asr/gigaspeech/data/train.csv |\
        cut -f 3 > examples/asr/gigaspeech/data/text
    # tools/kenlm/build/bin/lmplz -o 4 < examples/asr/gigaspeech/data/text \
    #     > examples/asr/aisehll/data/4gram.arpa || exit 1
    tail -n +2 examples/asr/gigaspeech/data/train.csv |\
        awk '{print $3"\t"$3}' > examples/asr/gigaspeech/data/train.trans.csv
    tail -n +2 examples/asr/gigaspeech/data/dev.csv |\
        awk '{print $3"\t"$3}' > examples/asr/gigaspeech/data/dev.trans.csv
    tail -n +2 examples/asr/gigaspeech/data/test.csv |\
        awk '{print $3"\t"$3}' > examples/asr/gigaspeech/data/test.trans.csv
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/asr/gigaspeech/configs/rnnlm.json || exit 1
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && [ "$use_wfst" = false ]; then
    # beam search decoding
    echo "Running decode with beam search..."
    python athena/inference.py \
        examples/asr/gigaspeech/configs/mtl_transformer_sp.json || exit 1
elif [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ] && [ "$use_wfst" = true ]; then
    # wfst decoding
    echo "Running decode with WFST..."
    echo "For now we will simply download LG.fst and words.txt from athena-decoder project"
    echo "Feel free to checkout graph creation manual at https://github.com/athena-team/athena-decoder#build-graph"
    git clone https://github.com/athena-team/athena-decoder
    cp athena-decoder/examples/gigaspeech/graph/LG.fst examples/asr/gigaspeech/data/
    cp athena-decoder/examples/gigaspeech/graph/words.txt examples/asr/gigaspeech/data/
    python athena/inference.py \
        examples/asr/gigaspeech/configs/mtl_transformer_sp_wfst.json || exit 1
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    # score-computing stage
    echo "computing score with sclite ..."
    if ! [ -x "$(command -v ./tools/SCTK/bin/sclite)" ]; then
        echo "sclite is not installed !!"
        exit 1
    else
        log=inference.log
        score_dir=score
        mkdir -p $score_dir

        python athena/tools/process_decode_result.py $log examples/asr/gigaspeech/data/vocab

        for file in $log.result $log.label; do
            awk '{print $0 "("NR".wav)"}' $file > $file.key
            mv $file.key $file
        done

        ./tools/SCTK/bin/sclite -i wsj -r $log.label -h $log.result -e utf-8 -o all -O $score_dir \
            > $score_dir/sclite.compute.log
        grep Err $score_dir/inference.log.result.sys
        grep Sum/Avg $score_dir/inference.log.result.sys
    fi
fi
