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
from absl import logging
import tqdm
from athena import SpeechRecognitionDatasetBuilder

def test():
    ''' test the speed of dataset '''
    dataset_builder = SpeechRecognitionDatasetBuilder(
        {
            "data_csv": "examples/asr/hkust/data/train.csv",
            "audio_config": {
                "type": "Fbank",
                "filterbank_channel_count": 40,
                "sample_rate": 8000,
                "local_cmvn": False,
            },
            "input_length_range": [200, 50000],
            "speed_permutation": [1.0],
            "text_config": {"type":"vocab", "model":"examples/asr/hkust/data/vocab"},
            "cmvn_file":"examples/asr/hkust/data/cmvn"
        }
    )
    dataset_builder.compute_cmvn_if_necessary(True)
    dataset = dataset_builder.as_dataset(16, 4)
    start = time.time()
    for _ in tqdm.tqdm(dataset, total=len(dataset_builder)//16):
        pass
    logging.info(time.time() - start)


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    test()
