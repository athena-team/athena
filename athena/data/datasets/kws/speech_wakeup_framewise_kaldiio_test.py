# -*- coding: utf-8 -*-
# Copyright (C) ATHENA AUTHORS; Yanguang Xu; Jianwei Sun
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
from absl import logging
import numpy as np
import os
from athena_wakeup import SpeechWakeupFramewiseDatasetKaldiIOBuilder
#np.set_printoptions(threshold=np.inf, suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test():
    dataset_builder = SpeechWakeupFramewiseDatasetKaldiIOBuilder(
        {
            "data_dir":"examples/kws/xhxh/data/test",
            "left_context":10,
            "right_context":10,
            "feat_dim":63
        }
    )
    dataset = dataset_builder.as_dataset(batch_size=32)
    for batch, item in enumerate(dataset):
        logging.info(item["input"].shape)
        logging.info(item["output"])
        logging.info(item["input_length"])
        logging.info(item["output_length"])
        if batch == 2: break

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    test()
