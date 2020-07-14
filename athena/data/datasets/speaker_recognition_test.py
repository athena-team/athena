# coding=utf-8
# Copyright (C) 2020 ATHENA AUTHORS; Ne Luo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=no-member, invalid-name
""" speaker dataset """

import time
import tqdm
from absl import logging
from athena import SpeakerRecognitionDatasetBuilder

def test():
    ''' test the speed of speaker dataset '''
    dataset_builder = SpeakerRecognitionDatasetBuilder(
        {
            "data_csv": "examples/speaker/voxceleb/data/vox1_test_wav.csv",
            "speaker_csv": "examples/speaker/voxceleb/data/vox1_test_wav.spk.csv",
            "audio_config": {
                "type": "Fbank",
                "filterbank_channel_count": 40,
                "sample_rate": 16000,
                "local_cmvn": False,
            },
            "cmvn_file":"examples/speaker/voxceleb/data/cmvn"
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
