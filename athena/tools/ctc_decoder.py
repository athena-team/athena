# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Chaoyang Mei
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
# Only support eager mode
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes

from typing import List, Tuple
from collections import defaultdict
import tensorflow as tf
import math


def log_add(args: List[float]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def ctc_prefix_beam_decoder(
        encoder_out: tf.Tensor,
        ctc_final_layer: tf.keras.layers.Dense,
        beam_size: int,
        blank_id: int,
) -> Tuple[List[Tuple[tuple, float]], tf.Tensor]:
    """ CTC prefix beam search inner implementation
        reference: https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/asr_model.py.
        Args:
            encoder_out (tf.Tensor): (batch, max_len, feat_dim)
            ctc_final_layer (tf.keras.layers.Dense)
            beam_size (int): beam size for beam search
            blank_id (int): blank id

        Returns:
            List[Tuple[tuple, float]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
    batch_size = encoder_out.shape[0]
    # For CTC prefix beam search, we only support batch_size=1
    # Let's assume B = batch_size and N = beam_size
    # 1. Encoder forward and get CTC score
    maxlen = encoder_out.shape[1]
    ctc_probs = tf.nn.log_softmax(ctc_final_layer(encoder_out), axis=2)  # (1, maxlen, vocab_size)
    ctc_probs = tf.squeeze(ctc_probs, axis=0)
    cur_hyps = [(tuple(), (0.0, -float('inf')))]
    # 2. CTC beam search step by step
    for t in range(0, maxlen):
        logp = ctc_probs[t]  # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        # 2.1 First beam prune: select topk best
        top_k_logp, top_k_index = tf.math.top_k(logp, k=beam_size)  # (beam_size,)
        for s in top_k_index:
            s = s.numpy().item()
            ps = logp[s].numpy().item()
            for prefix, (pb, pnb) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else None
                if s == blank_id:  # blank
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pb = log_add([n_pb, pb + ps, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                elif s == last:
                    #  Update *ss -> *s;
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pnb = log_add([n_pnb, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                    # Update *s-s -> *ss, - is for blank
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)
                else:
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)

        # 2.2 Second beam prune
        next_hyps = sorted(next_hyps.items(),
                           key=lambda x: log_add(list(x[1])),
                           reverse=True)
        cur_hyps = next_hyps[:beam_size]
    hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
    return hyps, encoder_out

def freeze_ctc_prefix_beam_decoder(
        encoder_out: tf.Tensor,
        ctc_final_layer: tf.keras.layers.Dense,
        beam_size: int,
        blank_id: int,
) -> Tuple[List[Tuple[tuple, float]], tf.Tensor]:
    """ CTC prefix beam search inner implementation
        reference: https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/asr_model.py.
        Args:
            encoder_out (tf.Tensor): (batch, max_len, feat_dim)
            ctc_final_layer (tf.keras.layers.Dense)
            beam_size (int): beam size for beam search
            blank_id (int): blank id

        Returns:
            List[Tuple[tuple, float]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
    batch_size = encoder_out.shape[0]
    # For CTC prefix beam search, we only support batch_size=1
    # Let's assume B = batch_size and N = beam_size
    # 1. Encoder forward and get CTC score
    maxlen = encoder_out.shape[1]
    ctc_probs = tf.nn.log_softmax(ctc_final_layer(encoder_out), axis=2)  # (1, maxlen, vocab_size)
    ctc_probs = tf.squeeze(ctc_probs, axis=0)
    cur_hyps = [(tuple(), (0.0, -float('inf')))]
    # 2. CTC beam search step by step
    for t in tf.range(0, maxlen):
        logp = ctc_probs[t]  # (vocab_size,)
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
        # 2.1 First beam prune: select topk best
        top_k_logp, top_k_index = tf.math.top_k(logp, k=beam_size)  # (beam_size,)
        for s in top_k_index:
            ps = logp[s]
            for prefix, (pb, pnb) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else None
                if s == blank_id:  # blank
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pb = log_add([n_pb, pb + ps, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                elif s == last:
                    #  Update *ss -> *s;
                    n_pb, n_pnb = next_hyps[prefix]
                    n_pnb = log_add([n_pnb, pnb + ps])
                    next_hyps[prefix] = (n_pb, n_pnb)
                    # Update *s-s -> *ss, - is for blank
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)
                else:
                    n_prefix = prefix + (s,)
                    n_pb, n_pnb = next_hyps[n_prefix]
                    n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                    next_hyps[n_prefix] = (n_pb, n_pnb)

        # 2.2 Second beam prune
        next_hyps = sorted(next_hyps.items(),
                           key=lambda x: log_add(list(x[1])),
                           reverse=True)
        cur_hyps = next_hyps[:beam_size]
    hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
    return hyps, encoder_out
