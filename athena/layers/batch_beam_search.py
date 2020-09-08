# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Miao Cao
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
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes

"""the batch beam search implementation"""
from collections import namedtuple
import tensorflow as tf

CandidateHolder = namedtuple(
    "CandidateHolder",
    ["cand_seqs", "cand_logits", "cand_scores", "cand_parents"],
)


class BatchBeamSearchLayer(tf.keras.layers.Layer):
    r""" Beam search decoding used in seq2seq decoder layer
    This layer is used for evaluation

    Args:
        num_syms: the size of the vocab
        sos: the representation of the start symbol, should be an id
        eos: the representation of the end symbol, should be an id
        beam_size: the beam size
    """

    def __init__(self, num_syms, sos, eos, beam_size, attention_dim, deocoder_dim):
        super().__init__()
        self.num_syms = num_syms
        self.sos = sos
        self.eos = eos
        self.beam_size = beam_size
        self.batch_size = None
        self.decoder_one_step = None
        self.attention_dim = attention_dim
        self.decoder_dim = deocoder_dim

    def set_one_step(self, decoder_one_step):
        """ Allocate the time propagating function of the decoder
        Args:
            decoder_one_step: the time propagating function of the decoder
        """
        self.decoder_one_step = decoder_one_step

    def beam_search_score(self, candidate_holder, decoder_states,
                          encoder_outputs, step, training=False):
        """Call the time propagating function, fetch the acoustic score at the current step
           If needed, call the auxiliary scorer and update cand_states in candidate_holder
        """
        cand_logits_array = candidate_holder.cand_logits
        cand_seqs_array = candidate_holder.cand_seqs
        logits, cand_logits_array, new_decoder_states = self.decoder_one_step(
            cand_logits_array, cand_seqs_array, step, decoder_states, encoder_outputs,
            training=training
        )
        cand_scores = candidate_holder.cand_scores.read(0)
        if step == 0:
            cand_scores = cand_scores[:, :1]
        Z = tf.reduce_logsumexp(logits, axis=(2,), keepdims=True)
        logprobs = logits - Z
        new_scores = logprobs + tf.expand_dims(cand_scores, 2)
        return new_scores, cand_logits_array, new_decoder_states

    def deal_with_completed(
        self,
        completed_scores_array,
        completed_seqs_array,
        completed_length_array,
        new_scores,
        candidate_holder,
        max_seq_len,
        step):
        """Add the new calculated completed seq with its score to completed seqs
           select top beam_size probable completed seqs with these corresponding scores
        """
        old_completed_scores = completed_scores_array.read(0)
        new_scores = new_scores[:, :, self.eos]
        new_completed_scores = tf.concat([old_completed_scores, new_scores], 1)

        cand_seqs = candidate_holder.cand_seqs.stack()
        if step == 0:
            cand_seqs = cand_seqs[:, :, :1]
        cand_seq_len = tf.shape(cand_seqs)[0]
        eos_tail = tf.fill(
            [self.batch_size,
             tf.shape(cand_seqs)[2],
             max_seq_len - cand_seq_len],
            self.eos,
        )
        new_completed_seqs = tf.concat(
            [tf.transpose(cand_seqs, [1, 2, 0]), eos_tail], axis=2
        )
        old_completed_seqs = tf.transpose(completed_seqs_array.stack(), [1, 2, 0])
        completed_seqs = tf.concat([old_completed_seqs, new_completed_seqs], 1)

        new_completed_length = tf.fill(
            [self.batch_size, tf.shape(new_completed_seqs)[1]], cand_seq_len + 1
        )
        old_completed_length = completed_length_array.read(0)
        completed_length = tf.concat([old_completed_length, new_completed_length], 1)
        completed_len = tf.shape(new_completed_scores)[1]
        if completed_len > self.beam_size:
            # Rescale scores by sequence length
            completed_length_float = tf.cast(completed_length, tf.float32)
            rescaled_scores = new_completed_scores / completed_length_float
            _, inds = tf.math.top_k(rescaled_scores, k=self.beam_size)
            inds = tf.reshape(inds, [self.batch_size, self.beam_size])
            new_completed_scores = tf.gather(new_completed_scores, inds, axis=1, batch_dims=1)
            completed_seqs = tf.gather(completed_seqs, inds, axis=1, batch_dims=1)
            completed_length = tf.gather(completed_length, inds, axis=1, batch_dims=1)
        completed_scores_array = completed_scores_array.write(0, new_completed_scores)
        completed_seqs_array = completed_seqs_array.unstack(tf.transpose(completed_seqs, [2, 0, 1]))
        completed_length_array = completed_length_array.write(0, completed_length)
        return completed_scores_array, completed_seqs_array, completed_length_array

    def deal_with_uncompleted(
        self,
        new_scores,
        new_cand_logits_array,
        candidate_holder,
        step,
        decoder_states
        ):
        """select top probable candidate seqs from new predictions with its scores
           update candidate_holder based on top probable candidates
        """

        cand_seqs = tf.transpose(candidate_holder.cand_seqs.stack(), [1, 2, 0])
        if step == 0:
            cand_seqs = cand_seqs[:, :1, :]
        num_cands = tf.shape(cand_seqs)[1]
        parents, syms = tf.meshgrid(
            tf.range(num_cands), tf.range(self.num_syms), indexing="ij"
        )
        parents = tf.tile(parents, [self.batch_size, 1])
        syms = tf.tile(syms, [self.batch_size, 1])
        parents = parents[:, :self.eos]
        syms = syms[:, :self.eos]

        new_scores_flat = tf.reshape(new_scores, [self.batch_size, -1])
        parents_flat = tf.reshape(parents, [self.batch_size, -1])
        syms_flat = tf.reshape(syms, [self.batch_size, -1])

        cand_scores, inds = tf.math.top_k(new_scores_flat, k=self.beam_size)

        cand_syms = tf.gather(syms_flat, inds, axis=1, batch_dims=1)
        cand_parents = tf.gather(parents_flat, inds, axis=1, batch_dims=1)

        new_cand_logits = tf.transpose(new_cand_logits_array.stack(), [1, 2, 0, 3])
        cand_logits = tf.gather(new_cand_logits, cand_parents, axis=1, batch_dims=1)
        cand_logits = tf.transpose(cand_logits, [2, 0, 1, 3])
        cand_seqs = tf.gather(cand_seqs, cand_parents, axis=1, batch_dims=1)

        (decoder_context, decoder_rnn1_states, decoder_rnn2_states) = decoder_states
        decoder_context = tf.reshape(decoder_context,
                                     [self.batch_size, -1, self.attention_dim])
        decoder_rnn1_states = tf.reshape(decoder_rnn1_states,
                                         [self.batch_size, -1, self.decoder_dim])
        decoder_rnn2_states = tf.reshape(decoder_rnn2_states,
                                         [self.batch_size, -1, self.decoder_dim])
        decoder_context = tf.gather(decoder_context, cand_parents, axis=1, batch_dims=1)
        decoder_rnn1_states = tf.gather(decoder_rnn1_states, cand_parents, axis=1, batch_dims=1)
        decoder_rnn2_states = tf.gather(decoder_rnn2_states, cand_parents, axis=1, batch_dims=1)
        decoder_context = tf.reshape(decoder_context, [-1, self.attention_dim])
        decoder_rnn1_states = tf.reshape(decoder_rnn1_states, [-1, self.decoder_dim])
        decoder_rnn2_states = tf.reshape(decoder_rnn2_states, [-1, self.decoder_dim])

        cand_syms = tf.expand_dims(cand_syms, 2)
        cand_seqs = tf.concat([cand_seqs, cand_syms], axis=2)
        cand_seqs = tf.transpose(cand_seqs, [2, 0, 1])

        cand_scores_array = candidate_holder.cand_scores
        cand_seqs_array = candidate_holder.cand_seqs
        cand_parents_array = candidate_holder.cand_parents
        cand_scores_array = cand_scores_array.write(0, cand_scores)
        cand_logits_array = new_cand_logits_array.unstack(cand_logits)
        cand_seqs_array = cand_seqs_array.unstack(cand_seqs)
        cand_parents_array = cand_parents_array.write(0, cand_parents)

        candidate_holder = CandidateHolder(
            cand_seqs_array, cand_logits_array, cand_scores_array, cand_parents_array
        )
        return candidate_holder, (decoder_context, decoder_rnn1_states, decoder_rnn2_states)

    def call(self, encoder_outputs, output_beam=1, training=False):

        self.batch_size = tf.shape(encoder_outputs[0])[0]

        max_seq_len = tf.shape(encoder_outputs[0])[1]

        cand_seqs_array = tf.TensorArray(dtype=tf.int32, size=1,
                                         dynamic_size=True, clear_after_read=False)
        cand_seqs_init = tf.ones([1, self.batch_size, self.beam_size], dtype=tf.int32) * self.eos
        cand_seqs_array = cand_seqs_array.unstack(cand_seqs_init)


        cand_logits_array = tf.TensorArray(dtype=tf.float32, size=1,
                                           dynamic_size=True, infer_shape=False)

        cand_scores_array = tf.TensorArray(dtype=tf.float32, size=1)
        cand_scores_init = tf.zeros([self.batch_size, self.beam_size], dtype=tf.float32)
        cand_scores_array = cand_scores_array.write(0, cand_scores_init)

        cand_parents_array = tf.TensorArray(dtype=tf.int32, size=1)
        cand_parents_init = tf.zeros([self.batch_size, self.beam_size], dtype=tf.int32)
        cand_parents_array = cand_parents_array.write(0, cand_parents_init)

        candidate_holder = CandidateHolder(
            cand_seqs_array, cand_logits_array, cand_scores_array, cand_parents_array
        )
        completed_seqs_array = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True)
        completed_seqs_init = tf.ones([max_seq_len, self.batch_size, self.beam_size],
                                      dtype=tf.int32) * self.eos
        completed_seqs_array = completed_seqs_array.unstack(completed_seqs_init)

        completed_length_array = tf.TensorArray(dtype=tf.int32, size=1)
        completed_length_init = tf.ones([self.batch_size, self.beam_size], dtype=tf.int32)
        completed_length_array = completed_length_array.write(0, completed_length_init)

        completed_scores_array = tf.TensorArray(dtype=tf.float32, size=1)
        completed_scores_init = tf.ones([self.batch_size, self.beam_size],
                                        dtype=tf.float32) * (-1e10)
        completed_scores_array = completed_scores_array.write(0, completed_scores_init)

        decoder_states = (tf.zeros([self.batch_size, self.attention_dim]),
                          tf.zeros([self.batch_size, self.decoder_dim]),
                          tf.zeros([self.batch_size, self.decoder_dim]))

        for pos in tf.range(50):
            # compute new scores
            new_scores, new_cand_logits_array, new_decoder_states = self.beam_search_score(
                candidate_holder, decoder_states, encoder_outputs, pos, training=training
            )
            # extract seqs with end symbol
            completed_scores_array, completed_seqs_array, completed_length_array = \
                self.deal_with_completed(
                    completed_scores_array,
                    completed_seqs_array,
                    completed_length_array,
                    new_scores,
                    candidate_holder,
                    max_seq_len,
                    pos
                )

            if (pos + 1) >= max_seq_len:
                break

            new_scores = new_scores[:, :, :self.eos]

            min_completed_score = tf.reduce_min(completed_scores_array.stack())
            max_new_score = tf.reduce_max(new_scores)
            #
            cand_seq_len = tf.shape(candidate_holder.cand_seqs.stack())[0] + 1
            cand_seq_len_float = tf.cast(cand_seq_len, tf.float32)
            new_scores_rescale = new_scores / cand_seq_len_float
            max_new_score_rescale = tf.reduce_max(new_scores_rescale)
            completed_length_float = tf.cast(completed_length_array.stack(), tf.float32)
            min_completed_score_rescale = tf.reduce_min(completed_scores_array.stack()
                                                        / completed_length_float)
            # Strict the statement of termination by scores and normalized scores(by length)
            if (
                max_new_score < min_completed_score
                or max_new_score_rescale < min_completed_score_rescale
            ) and pos > 2:
                break

            candidate_holder, decoder_states = self.deal_with_uncompleted(
                new_scores, new_cand_logits_array, candidate_holder, pos, new_decoder_states
            )
            if tf.shape(encoder_outputs[0])[0] < self.batch_size * self.beam_size:
                encoder_output, input_mask = encoder_outputs
                encoder_output_scale = tf.tile(encoder_output[:, tf.newaxis, :, :],
                                               [1, self.beam_size, 1, 1])
                input_mask_scale = tf.tile(input_mask[:, tf.newaxis, :, :, :],
                                           [1, self.beam_size, 1, 1, 1])
                encoder_output = tf.reshape(encoder_output_scale,
                                            [-1, tf.shape(encoder_output)[1], 512])
                input_mask = tf.reshape(input_mask_scale,
                                        [-1,
                                         tf.shape(input_mask)[1],
                                         tf.shape(input_mask)[2],
                                         tf.shape(input_mask)[3]])
                encoder_outputs = (encoder_output, input_mask)

        # Sort completed seqs
        completed_seqs = tf.transpose(completed_seqs_array.stack(), [1, 2, 0])
        completed_scores = completed_scores_array.read(0)
        completed_length = completed_length_array.read(0)

        completed_length_float = tf.cast(completed_length, tf.float32)
        rescaled_scores = completed_scores / completed_length_float

        max_length = tf.math.reduce_max(completed_length)
        result = completed_seqs[:, :, 1:max_length]
        result = tf.reshape(result, [self.batch_size*output_beam, -1])
        result = tf.cast(result, tf.int64)
        return result, rescaled_scores
