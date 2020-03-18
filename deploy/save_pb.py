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
import json
import tensorflow as tf
from absl import logging
from athena import *
import tensorflow.compat.v1 as tf1
import subprocess
import os

SUPPORTED_DATASET_BUILDER = {
    "speech_recognition_dataset": SpeechRecognitionDatasetBuilder,
    "speech_dataset": SpeechDatasetBuilder,
    "language_dataset": LanguageDatasetBuilder,
}

SUPPORTED_MODEL = {
    "deep_speech": DeepSpeechModel,
    "speech_transformer": SpeechTransformer,
    "speech_transformer2": SpeechTransformer2,
    "mtl_transformer_ctc": MtlTransformerCtc,
    "mpc": MaskedPredictCoding,
    "rnnlm": RNNLM,
    "translate_transformer": NeuralTranslateTransformer,
}

SUPPORTED_OPTIMIZER = {
    "warmup_adam": WarmUpAdam,
    "expdecay_adam": ExponentialDecayAdam,
    "adam": tf.keras.optimizers.Adam,
}

DEFAULT_CONFIGS = {
    "pb_path":"",
    "batch_size": 32,
    "num_epochs": 20,
    "sorta_epoch": 1,
    "ckpt": None,
    "summary_dir": None,
    "solver_gpu": [0],
    "solver_config": None,
    "model": "speech_transformer",
    "num_classes": None,
    "model_config": None,
    "pretrained_model": None,
    "optimizer": "warmup_adam",
    "optimizer_config": None,
    "num_data_threads": 1,
    "dataset_builder": "speech_recognition_dataset",
    "trainset_config": None,
    "devset_config": None,
    "testset_config": None,
    "decode_config": None,
}

def parse_config(config):
    """ parse config """
    p = register_and_parse_hparams(DEFAULT_CONFIGS, config, cls="main")
    logging.info("hparams: {}".format(p))
    return p

def freeze_session(session,keep_var_names=None,output_names=None,clear_devices=True):
    graph = session.graph
    with graph.as_default():
        output_names = output_names or []
        print("output_names",output_names)
        input_graph_def = graph.as_graph_def()
        print("len node1",len(input_graph_def.node))
    if clear_devices:
        for node in input_graph_def.node:
            node.device = ""
    frozen_graph =  tf1.graph_util.convert_variables_to_constants(session, input_graph_def, output_names)
    outgraph = tf1.graph_util.remove_training_nodes(frozen_graph)
    print("##################################################################")
    for node in outgraph.node:
        print('node:', node.name)
    print("len node1",len(outgraph.node))
    return outgraph

def save_pb(jsonfile, Solver, rank_size=1, rank=0):
    config = None
    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)

    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.trainset_config)
    if p.trainset_config is None and p.num_classes is None:
        raise ValueError("trainset_config and num_classes can not both be null")

    #set graph
    tf1.reset_default_graph()
    tf1.keras.backend.set_learning_phase(0)
    tf1.disable_v2_behavior()

    model = SUPPORTED_MODEL[p.model](
        data_descriptions=dataset_builder,
        config=p.model_config,
    )

    #restore from ckpt
    optimizer = SUPPORTED_OPTIMIZER[p.optimizer](p.optimizer_config)
    checkpointer = Checkpoint(
        checkpoint_directory=p.ckpt,
        model=model,
        optimizer=optimizer,
    )
    checkpointer.restore_from_best()
    #checkpointer.restore('examples/tts/m15/ckpts/merlin_aco/ckpt-19')

    frozen_graph = freeze_session(tf1.keras.backend.get_session(),output_names=[out.op.name for out in model.ks_encoder.outputs])
    deploy_path = p.pb_path
    if not os.path.exists(deploy_path):
        subprocess.getoutput('mkdir -p %s'%deploy_path)
    tf1.train.write_graph(frozen_graph, deploy_path, "encoder.pb", as_text=False)
    print(" successful save encoder as pb")

    frozen_graph = freeze_session(tf1.keras.backend.get_session(),output_names=[out.op.name for out in model.ks_decoder.outputs])
    tf1.train.write_graph(frozen_graph, deploy_path, "decoder.pb", as_text=False)
    print(" successful save decoder as pb")

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)

    json_file = sys.argv[1]
    config = None
    with open(json_file) as f:
        config = json.load(f)
    p = parse_config(config)
    BaseSolver.initialize_devices(p.solver_gpu)
    save_pb(json_file, BaseSolver, 1, 0)
    print("done!")
