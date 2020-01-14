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
dataset_dir=/nfs/project/datasets/opensource_data/hkust

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "Preparing data"
    mkdir -p examples/asr/hkust/data || exit 1
    python examples/asr/hkust/local/prepare_data.py \
        $dataset_dir examples/asr/hkust/data || exit 1

    # calculate cmvn
    cat examples/asr/hkust/data/train.csv > examples/asr/hkust/data/all.csv
    tail -n +2 examples/asr/hkust/data/dev.csv >> examples/asr/hkust/data/all.csv
    python athena/cmvn_main.py \
      examples/asr/hkust/mpc.json examples/asr/hkust/data/all.csv || exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # pretrain stage
    echo "Pretraining"
    python horovodrun -np 4 -H localhost:4 athena/horovod_main.py examples/asr/hkust/mpc.json || exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # finetuning stage
    echo "Fine-tuning"
    python horovodrun -np 4 -H localhost:4 athena/horovod_main.py examples/asr/hkust/mtl_transformer_sp.json || exit 1
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # decoding stage
    echo "Decoding"
    # prepare language model
    tail -n +2 examples/asr/hkust/data/train.csv |\
        cut -f 3 > examples/asr/hkust/data/text || exit 1
    python examples/asr/hkust/local/segment_word.py \
        examples/asr/hkust/data/vocab \
        examples/asr/hkust/data/text \
        > examples/asr/hkust/data/text.seg || exit 1
    tools/kenlm/build/bin/lmplz -o 5 < examples/asr/hkust/data/text.seg \
        > examples/asr/hkust/data/5gram.arpa || exit 1

    python athena/decode_main.py examples/asr/hkust/mtl_transformer.json || exit 1
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "training rnnlm"
    tail -n +2 examples/asr/hkust/data/train.csv | awk '{print $3"\t"$3}' > examples/asr/hkust/data/train.trans.csv
    tail -n +2 examples/asr/hkust/data/dev.csv | awk '{print $3"\t"$3}' > examples/asr/hkust/data/dev.trans.csv
    python athena/main.py examples/asr/hkust/rnnlm.json || exit 1
fi

