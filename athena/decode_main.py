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
# pylint: disable=invalid-name, no-member, redefined-outer-name
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
    p, model, _, checkpointer = build_model_from_jsonfile(jsonfile)
    lm_model = None
    if 'lm_type' in p.decode_config and p.decode_config['lm_type'] == "rnn":
        _, lm_model, _, lm_checkpointer = build_model_from_jsonfile(p.decode_config['lm_path'])
        lm_checkpointer.restore_from_best()
    checkpointer.restore_from_best()
    solver = DecoderSolver(model, config=p.decode_config, lm_model=lm_model)
    assert p.testset_config is not None
    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.testset_config)
    dataset_builder = dataset_builder.compute_cmvn_if_necessary(True)
    solver.decode(dataset_builder.as_dataset(batch_size=1))


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)

    jsonfile = sys.argv[1]
    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)

    DecoderSolver.initialize_devices(p.solver_gpu)
    decode(jsonfile)
