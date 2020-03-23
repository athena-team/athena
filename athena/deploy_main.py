# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Xiaoning Lei
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
""" a sample implementation of LAS for HKUST """
import sys
import tensorflow as tf
from absl import logging
import tensorflow.compat.v1 as tf1
import subprocess
import os
from athena.main import (
    build_model_from_jsonfile
)

def freeze_session(session, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        output_names = output_names or []
        logging.info("output_names: %s" % str(output_names))
        input_graph_def = graph.as_graph_def()
    if clear_devices:
        for node in input_graph_def.node:
            node.device = ""
    frozen_graph =  tf1.graph_util.convert_variables_to_constants(session, input_graph_def, output_names)
    outgraph = tf1.graph_util.remove_training_nodes(frozen_graph)
    return outgraph

def save_pb(jsonfile, deploy_path):
    #set graph
    tf1.reset_default_graph()
    tf1.keras.backend.set_learning_phase(0)
    tf1.disable_v2_behavior()

    p, model, _, checkpointer = build_model_from_jsonfile(jsonfile, pre_run=False)
    model.deploy()
    checkpointer.restore_from_best()

    if hasattr(model, "deploy_encoder") and model.deploy_encoder is not None:
        frozen_graph = freeze_session(tf1.keras.backend.get_session(),
                                      output_names=[out.op.name for out in model.deploy_encoder.outputs])
        tf1.train.write_graph(frozen_graph, deploy_path, "encoder.pb", as_text=False)
        logging.info("successful save encoder as pb")
    if hasattr(model, "deploy_decoder") and model.deploy_decoder is not None:
        frozen_graph = freeze_session(tf1.keras.backend.get_session(),
                                      output_names=[out.op.name for out in model.deploy_decoder.outputs])
        tf1.train.write_graph(frozen_graph, deploy_path, "decoder.pb", as_text=False)
        logging.info("successful save decoder as pb")

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        logging.warning('Usage: python {} config_json_file pb_path'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)
    json_file = sys.argv[1]
    pb_path = sys.argv[2]
    if not os.path.exists(pb_path):
        subprocess.getoutput('mkdir -p %s'%pb_path)
    save_pb(json_file, pb_path)
    logging.info("done!")
