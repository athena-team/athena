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
# pylint: disable=invalid-name
""" ctc scorer used in joint-ctc decoding """
import numpy as np
import tensorflow as tf

class CTCPrefixScorer:
    """
    ctc one pass decoding, the algorithm is based on
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    """

    def __init__(self, eos, ctc_beam, num_classes, blank=-1, ctc_weight=0.25):
        self.logzero = -10000000000.0
        self.eos = eos
        self.blank = blank
        self.ctc_beam = ctc_beam
        self.state_index = 0
        self.score_index = 0
        self.ctc_weight = ctc_weight
        self.num_classes = num_classes
        self.x = None
        self.input_length = None
        self.init_state = None

    def initial_state(self, init_cand_states, x):
        """
        Initialize states and Add init_state and init_score to init_cand_states

        Args:
            init_cand_states: CandidateHolder.cand_states
            x: log softmax value from ctc_logits, shape: (beam, T, num_classes)

        Return: 
            init_cand_states
        """
        self.x = x[0].numpy()
        self.input_length = len(self.x)
        r = np.full((self.input_length, 2), self.logzero, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank]
        for i in tf.range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        self.init_state = r
        init_cand_states.append(
            tf.convert_to_tensor(np.array([[r] * self.num_classes]))
        )
        self.state_index = len(init_cand_states) - 1
        init_cand_states.append(
            tf.convert_to_tensor(np.array([[0] * self.num_classes]))
        )
        self.score_index = len(init_cand_states) - 1
        return init_cand_states

    def score(self, candidate_holder, new_scores):
        """
        Call this function to compute the ctc one pass decoding score based on
        the logits of the ctc module, the scoring function shares a common interface

        Args:
            candidate_holder: CandidateHolder
            new_scores: the score from other models

        Return:
            ctc_score_result: shape: (beam, num_classes)
            cand_states: CandidateHolder.cand_states updated cand_states
        """
        cand_seqs = candidate_holder.cand_seqs
        cand_states = candidate_holder.cand_states
        cand_syms = cand_seqs[:, -1]

        cand_state_value = []
        cand_score_value = []
        for j in range(cand_states[self.state_index].shape[0]):
            cand_state_value.append(cand_states[self.state_index][j][cand_syms[j]])
            cand_score_value.append(cand_states[self.score_index][j][cand_syms[j]])
        ctc_score_result = []
        ctc_score_total = []
        new_states = []
        for i in tf.range(new_scores.shape[0]):
            num_sym_state = np.array([self.init_state] * self.num_classes)
            num_sym_score = np.array([0.0] * self.num_classes, dtype=np.float32)
            num_sym_score_minus = np.array([0.0] * self.num_classes, dtype=np.float32)
            cand_seq = cand_seqs[i]
            ctc_pre_state = cand_state_value[i]
            top_ctc_candidates = np.argsort(new_scores[i, :])
            top_ctc_candidates = sorted(top_ctc_candidates[-self.ctc_beam :].tolist())
            cand_seq = np.array(cand_seq)
            top_ctc_candidates = np.array(top_ctc_candidates)
            ctc_pre_state = ctc_pre_state.numpy()
            ctc_score, new_state = self.cand_score(
                cand_seq, top_ctc_candidates, ctc_pre_state
            )
            ctc_pre_score = tf.cast(cand_score_value[i], tf.float32)
            ctc_score_minus = self.ctc_weight * (ctc_score - ctc_pre_score) + 500

            for k in range(len(top_ctc_candidates)):
                num_sym_score[top_ctc_candidates[k]] = ctc_score[k]
                num_sym_score_minus[top_ctc_candidates[k]] = ctc_score_minus[k]
                num_sym_state[top_ctc_candidates[k]] = new_state[k]
            num_sym_score_minus -= 500
            ctc_score_result.append(num_sym_score_minus)
            ctc_score_total.append(num_sym_score)
            new_states.append(num_sym_state)
        cand_states[self.state_index] = tf.convert_to_tensor(np.array(new_states))
        ctc_score_result = tf.convert_to_tensor(np.array(ctc_score_result))
        ctc_score_total = tf.convert_to_tensor(np.array(ctc_score_total))
        cand_states[self.score_index] = ctc_score_total
        return ctc_score_result, cand_states

    def cand_score(self, y, cs, r_prev):
        """
        Args:
            y: cand_seq
            cs: top_ctc_candidates
            r_prev: ctc_pre_state
            r: the probability of the output seq containing the predicted label
                 given the current input seqs, shape: [input_length, 2, ctc_beam]
            r[:, 0]: the prediction of the t-th frame is not blank
            r[:, 1]: the prediction of the t-th frame is blank
            log_phi: the probability that the last predicted label is not created
                      by the t-th frame
            log_psi: the sum of all log_phi's, the prefix probability, shape:[ctc_beam]

        Return:
            log_psi: ctc_score
            new_state
        """
        output_length = len(y) - 1  # ignore sos

        r = np.ndarray((self.input_length, 2, len(cs)), dtype=np.float32)
        xs = self.x[:, cs]
        if output_length == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.logzero
        else:
            # output length larger than input length is not supported
            r[output_length - 1] = self.logzero
        # r_sum: creating (t-1) labels, shape:[input_length]
        r_sum = np.logaddexp(r_prev[:, 0], r_prev[:, 1])
        last = y[-1]
        if output_length > 0 and last in cs:
            log_phi = np.ndarray((self.input_length, len(cs)), dtype=np.float32)
            for i in tf.range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum

        start = max(output_length, 1)
        log_psi = r[start - 1, 0]
        for t in tf.range(start, self.input_length):
            r[t, 0] = np.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = np.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.x[t, self.blank]
            log_psi = np.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        eos_pos = np.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]

        return log_psi, np.rollaxis(r, 2)
