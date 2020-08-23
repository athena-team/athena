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
""" starting point for conversion of VC models """
import sys
import json
import tensorflow as tf
from absl import logging
from athena import ConvertSolver
from athena.stargan_main import (
    parse_config,
    build_model_from_jsonfile_stargan,
    SUPPORTED_DATASET_BUILDER
)


def convert(jsonfile):
    """ entry point for speech conversion, do some preparation work """
    p, model, checkpointer = build_model_from_jsonfile_stargan(jsonfile)
    checkpointer.restore_from_best()
    assert p.testset_config is not None
    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.testset_config)
    solver = ConvertSolver(model, dataset_builder, config=p.convert_config)
    solver.convert(dataset_builder.as_dataset(batch_size=1))


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

    ConvertSolver.initialize_devices(p.solver_gpu)
    convert(jsonfile)

