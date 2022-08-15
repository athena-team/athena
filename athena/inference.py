# coding=utf-8
# Copyright (C) ATHENA AUTHORS; Ruixiong Zhang; Jianwei Sun
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
# pylint: disable=invalid-name, no-member, redefined-outer-name
r""" entry point for inference of all kinds of models """
import sys
import json
import tensorflow as tf
from absl import logging, flags
from athena.main import (
    parse_config,
    build_model_from_jsonfile,
    SUPPORTED_DATASET_BUILDER
)
from athena import BaseSolver, HorovodSolver
from athena.main import SOLVERS


def inference(jsonfile, config, rank_size=1, rank=0):
    """ entry point for model inference, do some preparation work """
    p, checkpointer, train_dataset_builder = build_model_from_jsonfile(jsonfile)

    assert p.testset_config is not None
    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.testset_config)
    dataset_builder.shard(rank_size, rank)
    logging.info("shard result: %d" % len(dataset_builder))

    inference_solver = SOLVERS[p.solver_type]
    # model average config
    avg_num = p.inference_config.get("model_avg_num", inference_solver.default_config.get("model_avg_num", 1))
    sort_by = p.inference_config.get("sort_by", inference_solver.default_config.get("sort_by", None))
    sort_by_time = p.inference_config.get("sort_by_time", inference_solver.default_config.get("sort_by_time", False))
    reverse = p.inference_config.get("sort_reverse", inference_solver.default_config.get("sort_reverse", True))
    if avg_num > 0:
        checkpointer.compute_nbest_avg(avg_num, sort_by=sort_by, sort_by_time=sort_by_time, reverse=reverse)
    if p.saved_model_dir != None:
        checkpointer.model = tf.keras.models.load_model(p.saved_model_dir)
        solver = inference_solver(model=checkpointer.model, data_descriptions=dataset_builder, config=p.inference_config)
        solver.inference_saved_model(dataset_builder, conf=p, rank_size=rank_size)
    else:
        solver = inference_solver(model=checkpointer.model, data_descriptions=dataset_builder, config=p.inference_config)
        solver.inference(dataset_builder, conf=p, rank_size=rank_size)


if __name__ == "__main__":
    logging.use_absl_handler()
    flags.FLAGS.mark_as_parsed()
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)

    jsonfile = sys.argv[1]
    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)
    if p.log_dir != None:
        logging.get_absl_handler().python_handler.stream = open(p.log_dir, "w")
    rank_size = 1
    rank_index = 0
    try:
        import horovod.tensorflow as hvd
        hvd.init()
        rank_size = hvd.size()
        rank_index = hvd.rank()
    except ImportError:
        print("There is some problem with your horovod installation. \
               But it wouldn't affect single-gpu inference")
    if rank_size > 1:
        HorovodSolver.initialize_devices(p.solver_gpu)
    else:
        BaseSolver.initialize_devices(p.solver_gpu)

    inference(jsonfile, p, rank_size=rank_size, rank=rank_index)
