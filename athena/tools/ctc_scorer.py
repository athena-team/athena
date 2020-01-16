# Copyright (C) ATHENA AUTHORS; Ruixiong Zhang;
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
import numpy as np
import tensorflow as tf
import time


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
        self.state_select_index = 0
        self.ctc_weight = ctc_weight
        self.num_classes = num_classes
        self.x = None
        self.input_length = None

    def initial_state(self, init_cand_states, x):
        """
        Initialize states and Add init_score to init_cand_states
        Args:
            init_cand_states: CandidateHolder.cand_states
            x: log softmax value from ctc_logits, shape: (beam, T, num_classes)
        Return: init_cand_states
        """
        x = tf.cast(x, tf.float32)
        self.x = x[0]
        self.input_length = tf.shape(self.x)[0]
        r = np.full((self.input_length, 2), self.logzero, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank]
        for i in tf.range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        init_cand_states.append(
            tf.convert_to_tensor(np.array([[r] * self.ctc_beam]), tf.float32)
        )
        # shape: [beam_search_size, ctc_beam, input_length, 2]
        self.state_index = len(init_cand_states) - 1
        init_cand_states.append(
            tf.convert_to_tensor(np.array([[0] * self.ctc_beam]), tf.float32)
        )
        # shape: [beam_search_size, ctc_beam]
        self.score_index = len(init_cand_states) - 1
        init_cand_states.append(
            tf.convert_to_tensor(np.array([[0] * self.num_classes]), tf.int32)
        )
        # shape: [beam_search_size, num_classes]
        self.state_select_index = len(init_cand_states) - 1
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

        cand_state_value = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        cand_score_value = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for j in range(cand_states[self.state_index].shape[0]):
            select_index = cand_states[self.state_select_index][j][cand_syms[j]]
            cand_state_value.write(j, cand_states[self.state_index][j][select_index])
            cand_score_value.write(j, cand_states[self.score_index][j][select_index])
        cand_score_value_beam_list = cand_score_value.stack()
        ctc_score_beam_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        new_state_beam_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        top_ctc_candidates_beam_list = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

        beam_search_size = new_scores.shape[0]
        for i in tf.range(beam_search_size):

            cand_seq = cand_seqs[i]
            ctc_pre_state = cand_state_value.read(i)
            top_ctc_candidates = tf.argsort(new_scores[i, :])
            top_ctc_candidates = tf.sort(top_ctc_candidates[-self.ctc_beam :])
            top_ctc_candidates_beam_list.write(i, top_ctc_candidates)
            ctc_score, new_state = self.cand_score(
                cand_seq, top_ctc_candidates, ctc_pre_state
            )
            ctc_score_beam_list.write(i, ctc_score)
            new_state_beam_list.write(i, new_state)
        ctc_score_beam_list = ctc_score_beam_list.stack()
        new_state_beam_list = new_state_beam_list.stack()
        top_ctc_candidates_beam_list = top_ctc_candidates_beam_list.stack()

        cand_score_value_beam_list = tf.expand_dims(cand_score_value_beam_list, 1)
        ctc_score_minus = self.ctc_weight * (ctc_score_beam_list - cand_score_value_beam_list) + 500
        ctc_score_beam_list = ctc_score_beam_list[:, :self.ctc_beam+1] # shape: [beam_search_size, ctc_beam]
        # shape: [beam_search_size, ctc_beam]
        cand_index_parent, cand_index_syms = tf.meshgrid(tf.range(beam_search_size),
                                                         tf.range(self.ctc_beam),
                                                         indexing='ij')
        cand_index_parent = tf.reshape(cand_index_parent, [-1, 1])
        top_ctc_candidates_beam_list = tf.reshape(top_ctc_candidates_beam_list, [-1, 1])
        cand_index = tf.concat([cand_index_parent, top_ctc_candidates_beam_list], axis=1)
        cand_index = tf.cast(cand_index, tf.int64) # shape: [total_score_select_num, 2 dimensions]

        ctc_score_minus_value = ctc_score_minus[: , :self.ctc_beam+1]
        ctc_score_minus_value = tf.reshape(ctc_score_minus_value, [-1])
        num_sym_score_minus = tf.sparse.SparseTensor(cand_index,
                                                     ctc_score_minus_value,
                                                     [beam_search_size, self.num_classes])
        num_sym_score_minus = tf.sparse.to_dense(num_sym_score_minus)
        num_sym_score_minus -= 500
        # shape: [beam_search_size, ctc_beam, input_length, 2]
        new_state_value = new_state_beam_list[:, :self.ctc_beam+1]
        cand_index_syms = tf.reshape(cand_index_syms, [-1])
        cand_state_score_index = tf.sparse.SparseTensor(cand_index,
                                                  cand_index_syms,
                                                  [beam_search_size, self.num_classes])
        cand_state_score_index = tf.sparse.to_dense(cand_state_score_index)

        cand_states[self.state_index] = new_state_value
        cand_states[self.score_index] = ctc_score_beam_list
        cand_states[self.state_select_index] = cand_state_score_index

        return num_sym_score_minus, cand_states

    def cand_score(self, y, cs, r_prev):
        """
        r: the probability of the current output seq produced by
           the current input seqs, shape: [input_length, 2, ctc_beam]
        r[:, 0]: the prediction of the t-th frame is not blank
        r[:, 1]: the prediction of the t-th frame is blank
        log_phi: when N frames create T outputs, it presents the probability that
                 the first N-1 frames create the first T-1 outputs
                 shape:[x_input_length, ctc_beam_size]
        log_psi: the prefix probability, shape:[ctc_beam]

        Args:
            y: cand_seq shape: [output_length]
            cs: top_ctc_candidates shape: [ctc_beam]
            r_prev: ctc_pre_state shape: [x_input_length, 2]
        Return:
            log_psi: ctc_score
            new_state
        """
        # output_length: the length of historical sequence excluding sos
        output_length = tf.shape(y)[0] - 1
        xs = tf.gather(self.x, cs, axis=1)
        xs_exp = tf.math.exp(xs)
        x_exp = tf.math.exp(self.x)
        # r_sum: creating (t-1) labels, shape:[input_length]
        r_sum = tf.math.reduce_logsumexp(r_prev, axis=1)
        last = y[-1]
        if output_length > 0 and tf.reduce_any(tf.equal(last, cs)):
            log_phi = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)
            for i in tf.range(tf.shape(cs)[0]):
                i_r_sum = r_sum
                if cs[i] == last:
                    i_r_sum = r_prev[:, 1]
                log_phi = log_phi.write(i, i_r_sum)
            log_phi = tf.transpose(log_phi.stack(), [1, 0])
        else:
            ctc_beam = tf.shape(cs)[0]
            log_phi = tf.tile(tf.expand_dims(r_sum, 1), [1, ctc_beam])
        start = tf.maximum(output_length, 1)

        r_nb = []
        r_b = []
        if output_length == 0:
            zeros = np.zeros([tf.shape(cs)[0]])
            r_nb.append(np.exp(xs[0].numpy()))
            r_b.append(zeros)
        else:
            # output length larger than input length is not supported
            r_first = np.zeros([output_length, 2, tf.shape(cs)[0]])
            r_nb.extend(r_first[:, 0, :])
            r_b.extend(r_first[:, 1, :])
        log_psi_exp = r_nb[start-1] # shape: [ctc_beam]
        r_nb_expand_exp = log_psi_exp # shape: [ctc_beam]
        log_phi_exp = tf.math.exp(log_phi)

        log_phi_exp, r_nb, r_b = self.loop(start, log_phi_exp, r_nb_expand_exp, xs_exp, x_exp, r_nb, r_b)
        log_phi_exp = log_phi_exp[:-1]
        xs_exp = xs_exp[1:]
        log_psi_exp_delta = tf.reduce_sum(log_phi_exp * xs_exp, axis=0) # shape: [ctc_beam]
        log_psi_exp = log_psi_exp + log_psi_exp_delta
        log_psi = tf.math.log(log_psi_exp)
        r_b = tf.expand_dims(r_b, 1)
        r_nb = tf.expand_dims(r_nb, 1)
        r = tf.math.log(tf.concat([r_nb, r_b], axis=1)) # shape: [length, 2, ctc_beam]
        eos_pos = tf.where(cs == self.eos)[:, 0]
        if tf.shape(eos_pos)[0] > 0:
            eos_pos = eos_pos[0]
            log_psi = tf.concat([log_psi[:eos_pos], r_sum[-1:], log_psi[eos_pos+1:]], axis=0)

        return log_psi, tf.transpose(r, [2, 0, 1])

    def loop(self, start, log_phi_exp, r_nb_expand_exp, xs_exp, x_exp, r_nb, r_b):
        log_phi_exp = log_phi_exp.numpy()
        r_nb_expand_exp = r_nb_expand_exp
        xs_exp = xs_exp.numpy()
        x_exp = x_exp.numpy()
        for t in range(start, self.input_length):
            log_phi_expand = log_phi_exp[t - 1] # shape: [ctc_beam]
            # not_blank shape: [1, ctc_beam]
            r_nb_expand_exp_tmp = (r_nb_expand_exp + log_phi_expand) * xs_exp[t]
            r_nb.append(r_nb_expand_exp_tmp)
            # blank shape: [1, ctc_beam]
            r_b.append((r_nb_expand_exp + r_b[t - 1]) * x_exp[t, self.blank])
            # the probability that the output can be produced by at least N frames
            #log_psi_exp = log_psi_exp + (log_phi_expand * xs_exp[t])
            r_nb_expand_exp = r_nb_expand_exp_tmp
        return tf.convert_to_tensor(log_phi_exp, dtype=tf.float32), \
               tf.convert_to_tensor(np.array(r_nb), dtype=tf.float32), \
               tf.convert_to_tensor(np.array(r_b), dtype=tf.float32)
