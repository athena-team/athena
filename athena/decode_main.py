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
import sys, os
import json
import tensorflow as tf
from absl import logging
from athena import DecoderSolver
from athena.main import (
    parse_config,
    build_model_from_jsonfile,
    SUPPORTED_DATASET_BUILDER
)


def decode(jsonfile, n=1, log_file=None):
    """ entry point for model decoding, do some preparation work """
    p, model, _, checkpointer = build_model_from_jsonfile(jsonfile)
    lm_model = None
    if 'lm_type' in p.decode_config and p.decode_config['lm_type'] == "rnn":
        _, lm_model, _, lm_checkpointer = build_model_from_jsonfile(p.decode_config['lm_path'])
        lm_checkpointer.restore_from_best()
    if not os.path.exists(log_file):
        raise IOError('The file is not exist！')
    checkpoint_wer_dict = {}
    for line in open(log_file):
        if 'epoch:' in line:
            splits = line.strip().split('\t')
            epoch = int(splits[0].split(' ')[-1])
            ctc_acc = float(splits[-1].split(' ')[-1])
            checkpoint_wer_dict[epoch] = ctc_acc
    checkpoint_wer_dict = {k: v for k, v in sorted(checkpoint_wer_dict.items(), key=lambda item: item[1], reverse=True)}
    ckpt_index_list = list(checkpoint_wer_dict.keys())[0: n]
    print('best_wer_checkpoint: ')
    print(ckpt_index_list)
    ckpt_v_list = []
    #restore v from ckpts
    for idx in ckpt_index_list:
        ckpt_path = p.ckpt + 'ckpt-' + str(idx + 1)
        checkpointer.restore(ckpt_path) #current variables will be updated
        var_list = []
        for i in model.trainable_variables:
            v = tf.constant(i.value())
            var_list.append(v)
        ckpt_v_list.append(var_list)
    #compute average, and assign to current variables
    for i in range(len(model.trainable_variables)):
        v = [tf.expand_dims(ckpt_v_list[j][i],[0]) for j in range(len(ckpt_v_list))]
        v = tf.reduce_mean(tf.concat(v,axis=0),axis=0)
        model.trainable_variables[i].assign(v)
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
    decode(jsonfile, n=5, log_file='nohup.out')
