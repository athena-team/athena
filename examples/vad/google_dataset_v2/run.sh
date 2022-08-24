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

set -e

# train options
stage=0
stop_stage=3

# horovod options

horovod_cmd="horovodrun -np 1 -H localhost:1"
horovod_prefix="horovod_"

# data options
dataset_dir=examples/vad/google_dataset_v2/data/resource
speech_dataset=google_speech_recognition_v2
background_dataset=rir_noises
speech_url="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
background_url="https://www.openslr.org/resources/28/rirs_noises.zip"

# source some path
. ./tools/env.sh

if [ "athena" != $(basename "$PWD") ]; then
    echo "You should run this script in athena directory!!"
    exit 1
fi

# download data source
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then

    echo "Downloading speech data"
    bash examples/vad/google_dataset_v2/local/download_data.sh $dataset_dir $speech_dataset $speech_url
    echo "Downloading background data"
    bash examples/vad/google_dataset_v2/local/download_data.sh $dataset_dir $background_dataset $background_url 
fi

# prepare data for athena training
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    
    echo "Preparing data"
    bash examples/vad/google_dataset_v2/local/prepare_data.sh $dataset_dir/$speech_dataset $dataset_dir/$background_dataset/RIRS_NOISES examples/vad/google_dataset_v2/data
fi

# training stage
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    echo "Training Model"
    python athena/main.py examples/vad/google_dataset_v2/configs/vad_dnn.json || exit 1
fi

# decode
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

    echo "Decoding"
    python athena/inference.py examples/vad/google_dataset_v2/configs/vad_dnn.json || exit 1
fi

echo "$0: Finished GOOGLE_DATASET_V2 training examples"
