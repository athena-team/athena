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
# Only support tensorflow 2.0
# pylint: disable=invalid-name, no-member
r""" a sample implementation of LAS for HKUST """
import sys
import json
from absl import logging
from athena.main import parse_config, SUPPORTED_DATASET_BUILDER
import os

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        logging.warning('Usage: python {} config_json_file data_csv_file'.format(sys.argv[0]))
        sys.exit()
    jsonfile, csv_file = sys.argv[1], sys.argv[2]

    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)
    if "speed_permutation" in p.trainset_config:
        p.dataset_config['speed_permutation'] = [1.0]
    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.trainset_config)

    if 'compute_cmvn_parallelly' in p:
        compute_cmvn_parallelly = p.compute_cmvn_parallelly
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        compute_cmvn_parallelly = False
    dataset_builder.load_csv(csv_file).compute_cmvn_if_necessary(True,compute_cmvn_parallelly)
