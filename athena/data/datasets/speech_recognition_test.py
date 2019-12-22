# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li
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
# pylint: disable=no-member, invalid-name
""" audio dataset """
import time
import tqdm
from absl import logging
from athena import SpeechRecognitionDatasetBuilder

def test():
    ''' test the speed of dataset '''
    data_csv = "/tmp-data/dataset/opensource/hkust/train.csv"
    dataset_builder = SpeechRecognitionDatasetBuilder(
        {
            "audio_config": {
                "type": "Fbank",
                "filterbank_channel_count": 40,
                "sample_rate": 8000,
                "local_cmvn": False,
            },
            "speed_permutation": [0.9, 1.0],
            "vocab_file": "examples/asr/hkust/data/vocab"
        }
    )
    dataset = dataset_builder.load_csv(data_csv).as_dataset(16, 4)
    start = time.time()
    for _ in tqdm.tqdm(dataset, total=len(dataset_builder)//16):
        pass
    logging.info(time.time() - start)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    test()
