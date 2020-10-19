# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li
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
r""" entry point for multi-gpu/ multi-machine training """
import sys
import json
import tensorflow as tf
import horovod.tensorflow as hvd
from absl import logging
from athena import HorovodSolver
from athena.main import parse_jsonfile, train


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)

    jsonfile = sys.argv[1]
    p = parse_jsonfile(jsonfile)
    HorovodSolver.initialize_devices(p.solver_gpu)
    #multi-servers training should use hvd.rank()
    train(jsonfile, HorovodSolver, hvd.size(), hvd.rank())

