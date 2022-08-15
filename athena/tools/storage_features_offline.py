# Copyright (C) 2022 ATHENA AUTHORS; Yanguang Xu
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
# pylint: disable=invalid-name
"""storage features offline"""
import sys
import json
from absl import logging

import tensorflow as tf
from athena import SpeechRecognitionDatasetKaldiIOBuilder, SpeechRecognitionDatasetBuilder
from athena.main import parse_config, SUPPORTED_DATASET_BUILDER

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    jsonfile = sys.argv[1]

    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    dataset_builder_class = SpeechRecognitionDatasetBuilder
    if p.trainset_config is not None:
        dataset_builder = dataset_builder_class(p.trainset_config)
        dataset_builder.storage_features_offline()
    else:
        logging.warning("There is no train config")
    if p.devset_config is not None:
        dataset_builder = dataset_builder_class(p.devset_config)
        dataset_builder.storage_features_offline()
    else:
        logging.warning("There is no dev config")
    if p.testset_config is not None:
        dataset_builder = dataset_builder_class(p.testset_config)
        dataset_builder.storage_features_offline()
    else:
        logging.warning("There is no test config")

