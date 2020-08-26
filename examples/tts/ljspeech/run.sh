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

dataset_dir=examples/tts/ljspeech/data

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "Creating csv"
    mkdir -p $dataset_dir || exit 1
    python examples/tts/ljspeech/local/prepare_data.py \
        $dataset_dir || exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] ; then
    # Training stage
    echo "Training"
    python athena/main.py \
        examples/tts/ljspeech/configs/t2.json
        > LJSpeech_training.log 2>&1 || exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] ; then
    # Inferencing stage
    echo "Inferencing"
    python athena/inference.py  \
        examples/tts/ljspeech/configs/t2.json
        > LJSpeech_synthesizing.log 2>&1 || exit 1
fi
