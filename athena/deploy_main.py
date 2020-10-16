# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Xiaoning Lei; Ne Luo
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
# pylint: disable=invalid-name, no-member, wildcard-import, unused-wildcard-import, redefined-outer-name
""" a sample implementation of freezing model """

import os
import sys
import json
import tensorflow as tf
from absl import logging
from athena import *
from athena.main import (
    parse_config,
    build_model_from_jsonfile
)


def freeze_session(session, output_names=None, clear_devices=True):
    """ build frozen graph """
    graph = session.graph
    with graph.as_default():
        output_names = output_names or []
        logging.info("output_names: %s" % str(output_names))
        input_graph_def = graph.as_graph_def()

    if clear_devices:
        for node in input_graph_def.node:
            node.device = ""
    frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                   session, input_graph_def, output_names)
    outgraph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)
    return outgraph


def freeze_graph_tf1(json_file, frozen_graph_dir):
    """ freeze TensorFlow model trained on Python to pb format,
        which gather graph defination and weights together into
        a single file .
    """
    # restore the best model
    _, model, _, checkpointer = build_model_from_jsonfile(json_file, pre_run=False)
    model.deploy()
    checkpointer.restore_from_best()
    model.save_weights(os.path.join(frozen_graph_dir, "model.h5"))

    tf.keras.backend.clear_session()
    tf.compat.v1.disable_eager_execution()

    _, model, _, _ = build_model_from_jsonfile(json_file, pre_run=False)
    model.deploy()
    model.load_weights(os.path.join(frozen_graph_dir, "model.h5"))

    session = tf.compat.v1.keras.backend.get_session()
    frozen_graph_enc = freeze_session(session, output_names=
                                      [out.op.name for out in model.deploy_encoder.outputs])
    tf.compat.v1.train.write_graph(frozen_graph_enc, frozen_graph_dir, "encoder.pb", as_text=False)

    frozen_graph_dec = freeze_session(session, output_names=
                                      [out.op.name for out in model.deploy_decoder.outputs])
    tf.compat.v1.train.write_graph(frozen_graph_dec, frozen_graph_dir, "decoder.pb", as_text=False)


def freeze_saved_model(json_file, frozen_graph_dir):
    """ freeze TensorFlow model trained on Python to saved model,
    """
    _, model, _, checkpointer = build_model_from_jsonfile(json_file, pre_run=False)

    def inference(x):
        samples = {"input": x}
        outputs = model.synthesize(samples)
        return outputs[0]

    model.deploy_function = tf.function(inference,
                                        input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    tf.saved_model.save(obj=model, export_dir=frozen_graph_dir)


FreezeFunctions = {
    "asr": freeze_graph_tf1,
    "tts": freeze_saved_model
}

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        logging.warning('Usage: python {} json_file frozen_graph_dir'.format(sys.argv[0]))
        sys.exit()

    tf.random.set_seed(1)
    json_file = sys.argv[1]
    with open(json_file) as f:
        config = json.load(f)
    p = parse_config(config)
    BaseSolver.initialize_devices(p.solver_gpu)
    frozen_graph_dir = sys.argv[2]
    if not os.path.exists(frozen_graph_dir):
        os.mkdir(frozen_graph_dir)

    freeze_function = FreezeFunctions[p.solver_type]
    freeze_function(json_file, frozen_graph_dir)
