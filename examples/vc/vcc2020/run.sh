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
        examples/vc/vcc2020/configs/stargan_voice_conversion.json examples/vc/vcc2020/data/all.csv || exit 1
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # training stage
    echo "training stargan"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/vc/vcc2020/configs/stargan_voice_conversion.json || exit 1
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # convert stage
    echo "convert wavs ..."
    python athena/convert_wavs.py \
        examples/vc/vcc2020/configs/stargan_voice_conversion.json || exit 1
fi
