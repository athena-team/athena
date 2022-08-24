# coding=utf-8
# Copyright (C) 2022 ATHENA AUTHORS
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
import sys
import time
import json
import numpy as np
import tensorflow as tf
from absl import logging, flags
from athena.main import (
    parse_config,
    build_model_from_jsonfile,
    SUPPORTED_DATASET_BUILDER
)
from athena import DecoderSolver
from athena import register_and_parse_hparams


def insert_blank(label, blank_id=0):
    """Insert blank token between every two label token."""
    label = np.expand_dims(label[0], 1) # batch size == 1
    blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
    label = np.concatenate([blanks, label], axis=1)
    label = label.reshape(-1)
    label = np.append(label, label[0])
    return label

def force_align(probs, y, blank_id=0):
    """ Force alignment 
    Args:
        ctc_probs: hidden state sequence, 3d tensor (B,T, D)
        y: id sequence tensor 2d tensor (B,L)
        int blank_id: blank symbol index
    Returns:
        alignment result (T)
    """
    y_insert_blank = insert_blank(y, blank_id=blank_id)
    log_alpha = np.zeros([tf.shape(probs)[1], len(y_insert_blank)]) - np.inf
    state_path = np.zeros([tf.shape(probs)[1], len(y_insert_blank)], dtype=np.int32) - 1
    log_alpha[0,0] = probs[0,0,y_insert_blank[0]]
    log_alpha[0,1] = probs[0,0,y_insert_blank[1]]
    for t in range(1, len(log_alpha)):
        for s in range(len(y_insert_blank)):
            if s == 0:
                candidates = np.array([log_alpha[t-1,s]])
                prev_state = np.array([s])
            elif y_insert_blank[s] == blank_id or s < 2 or y_insert_blank[s] == y_insert_blank[s-2]:
                candidates = np.array([log_alpha[t-1,s], log_alpha[t-1,s-1]])
                prev_state = np.array([s, s-1])
            else:
                candidates = np.array([log_alpha[t-1,s], log_alpha[t-1,s-1], log_alpha[t-1,s-2]])
                prev_state = np.array([s, s-1, s-2])
            log_alpha[t,s] = np.max(candidates) + probs[0, t, y_insert_blank[s]]
            state_path[t,s] = prev_state[np.argmax(candidates)]
    
    state_seq = -1 * np.ones(len(log_alpha), dtype=np.int32)
    candidates = np.array([log_alpha[-1, len(y_insert_blank)-1], log_alpha[-1, len(y_insert_blank)-2]])
    prev_state = [len(y_insert_blank) - 1, len(y_insert_blank) - 2]
    state_seq[-1] = prev_state[np.argmax(candidates)]

    for t in range(len(log_alpha)-2, -1, -1):
        state_seq[t] = state_path[t+1, state_seq[t+1]]
    alignmants = np.zeros(len(log_alpha), dtype=np.int32)
    for t in range(len(log_alpha)):
        alignmants[t] = y_insert_blank[state_seq[t]]
    return alignmants

def alignment(jsonfile, config, rank_size=1, rank=0):
    """ CTC Alignmnet according encoder probs """
    default_config = {
        "sampling_rate": 4,
        "align_out_file": "examples/asr/qianyue/ALIGN",
        "model_avg_num": 1
    }
    p, checkpointer, train_dataset_builder = build_model_from_jsonfile(jsonfile)
    assert p.batch_size == 1
    avg_num = 1 if 'model_avg_num' not in p.alignment_config else p.alignment_config['model_avg_num']
    if avg_num > 0:
        checkpointer.compute_nbest_avg(avg_num)
    assert p.testset_config is not None
    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.testset_config)
    
    blank_id = dataset_builder.num_class
    print('blank_id: {}'.format(blank_id))
    
    dataset_builder.shard(rank_size, rank)
    logging.info("shard result: %d" % len(dataset_builder))
    dataset = dataset_builder.as_dataset(batch_size=config.batch_size)
    
    hparams = register_and_parse_hparams(default_config, p.alignment_config)

    with open(hparams.align_out_file, "w") as align_file_out:
        for idx, samples in enumerate(dataset):
            begin = time.time()
            samples = checkpointer.model.prepare_samples(samples)
            speech = samples['input']
            speech_length = samples['input_length']
            label = samples['output']
            utt = samples['utt_id'][0].numpy().decode()
            speech_length = checkpointer.model.compute_logit_length(speech_length)
            encoder_ctc_probs, _ = checkpointer.model._forward_encoder_log_ctc(samples, False)
            alignments = force_align(encoder_ctc_probs, label, blank_id=blank_id)
            alignments = np.repeat(alignments, hparams.sampling_rate)
            align_file_out.write('{}\t{}\n'.format(utt, alignments))


if __name__ == "__main__":
    logging.use_absl_handler()
    flags.FLAGS.mark_as_parsed()
    logging.get_absl_handler().python_handler.stream = open("alignment.log", "w")
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)

    jsonfile = sys.argv[1]
    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)

    alignment(jsonfile, p, rank_size=1, rank=0)
