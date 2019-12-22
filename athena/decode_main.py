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
import tensorflow as tf
from absl import logging
from athena import DecoderSolver
from athena.main import (
    parse_config,
    build_model_from_jsonfile,
    SUPPORTED_DATASET_BUILDER
)


def decode(jsonfile):
    """ entry point for model decoding, do some preparation work """
    p, model, _, checkpointer, _ = build_model_from_jsonfile(jsonfile, 0)
    if "speed_permutation" in p.dataset_config:
        p.dataset_config['speed_permutation'] = [1.0]
    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.dataset_config)
    checkpointer.restore_from_best()
    solver = DecoderSolver(model, config=p.decode_config)
    dataset_builder = dataset_builder.load_csv(p.test_csv).compute_cmvn_if_necessary(True)
    solver.decode(dataset_builder.as_dataset(batch_size=1))


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    tf.random.set_seed(1)

    jsonfile = sys.argv[1]
    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)
    DecoderSolver.initialize_devices(p.solver_gpu)
    decode(jsonfile)
