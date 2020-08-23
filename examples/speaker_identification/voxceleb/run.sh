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
stop_stage=0
horovod_cmd=""
horovod_prefix=""
#horovod_cmd="horovodrun -np 4 -H localhost:4"
#horovod_prefix="horovod_"

# the user and password should be applied manually according to README.md
user=''
password=''

# please modify the following path accordingly 
# if you put dataset in the differenct path
data_dir=examples/speaker_identification/voxceleb/data

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    echo "Preparing data"
    mkdir -p examples/speaker_identification/voxceleb/data || exit 1
    python examples/speaker_identification/voxceleb/local/prepare_data.py \
        $data_dir $user $password || exit 1

    # seperate training and dev set
    cd $data_dir
    wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt

    tail -n +2 vox1_test_wav.csv > vox1.csv
    tail -n +2 vox1_dev_wav.csv >> vox1.csv
    grep "3 " iden_split.txt | awk -F " " '{print $2}' > iden_split_test.txt
    grep -v "3 " iden_split.txt | awk -F " " '{print $2}' > iden_split_train.txt

    while IFS= read -r line; do
        grep "$line" vox1.csv >> vox1_iden_test.csv
    done < iden_split_test.txt

    comm -3 <(sort vox1.csv) <(sort vox1_iden_test.csv) > vox1_iden_train.csv
    sed -i '1s/^/wav_filename\twav_length_ms\tspeaker_id\tspeaker_name\n/' \
        vox1_iden_test.csv vox1_iden_train.csv
    cd -
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # calculate cmvn
     echo "Calculating cmvn"
    python athena/cmvn_main.py \
        examples/speaker_identification/voxceleb/configs/speaker_resnet.json \
        examples/speaker_identification/voxceleb/data/vox1_iden_train.csv || exit 1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # training stage
    echo "Training"
    $horovod_cmd python athena/${horovod_prefix}main.py \
        examples/speaker_identification/voxceleb/configs/speaker_resnet.json || exit 1
fi
