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
dataset_dir=examples/tts/libritts/data

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "Creating csv"
    mkdir -p $dataset_dir
    python examples/tts/libritts/local/prepare_data.py \
        $dataset_dir || exit 1
fi