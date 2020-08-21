# Copyright (C) ATHENA AUTHORS; Ruixiong Zhang; Dongwei Jiang
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
""" the beam search decoder layer in encoder-decoder models """
from collections import namedtuple
import tensorflow as tf
from .ctc_scorer import CTCPrefixScorer
from .lm_scorer import NGramScorer, RNNScorer

CandidateHolder = namedtuple(
    "CandidateHolder",
    ["cand_seqs", "cand_logits", "cand_states", "cand_scores", "cand_parents"],
)


class BeamSearchDecoder:
    """Beam search decoding used in seq2seq decoder layer
    """

    def __init__(self, num_class, sos, eos, beam_size):
        """
        Args:
            num_class: the size of the vocab
            sos: start symbol, should be an id
            eos: end symbol, should be an id
            beam_size: the beam size
        """
        self.num_class = num_class
        self.sos = sos
        self.eos = eos
        self.beam_size = beam_size
        self.scorers = []
        self.ctc_scorer = None
        self.lm_model = None
        self.states = []
        self.decoder_one_step = None

    @staticmethod
    def build_decoder(hparams, num_class, sos, eos, decoder_one_step, lm_model=None):
        """Allocate the time propagating function of the decoder, initialize the decoder
        
        Args:
            hparams: the decoding configs are included here
            num_class: the size of the vocab
            sos: the start symbol index
            eos: the end symbol index
            decoder_one_step: the time propagating function of the decoder
            lm_model: the initialized language model
        Returns:
            beam_search_decoder: the initialized beam search decoder
        """
        beam_size = 1 if not hparams.decoder_type == 'beam_search_decoder' else hparams.beam_size
        beam_search_decoder = BeamSearchDecoder(
            num_class, sos, eos, beam_size=beam_size
        )
        beam_search_decoder.decoder_one_step = decoder_one_step
        if hparams.decoder_type == 'beam_search_decoder' and hparams.ctc_weight != 0:
            ctc_scorer = CTCPrefixScorer(
                eos,
                ctc_beam=hparams.beam_size * 2,
                num_classes=num_class,
                ctc_weight=hparams.ctc_weight,
            )
            beam_search_decoder.set_ctc_scorer(ctc_scorer)
        if hparams.lm_weight != 0:
            if hparams.lm_type == "ngram":
                if hparams.lm_path is None:
                    raise ValueError("lm path should not be none")
                lm_scorer = NGramScorer(
                    hparams.lm_path,
                    sos,
                    eos,
                    num_class,
                    lm_weight=hparams.lm_weight,
                )
                beam_search_decoder.set_lm_model(lm_scorer)
            elif hparams.lm_type == "rnn":
                lm_scorer = RNNScorer(
                    lm_model,
                    lm_weight=hparams.lm_weight)
                beam_search_decoder.set_lm_model(lm_scorer)
        return beam_search_decoder

    def set_lm_model(self, lm_model):
        """ set the lm_model

        Args:
            lm_model: lm_model
        """
        self.lm_model = lm_model
        self.scorers.append(lm_model)

    def set_ctc_scorer(self, ctc_scorer):
        """ set the ctc_scorer

        Args:
            ctc_scorer: the ctc scorer
        """
        self.ctc_scorer = ctc_scorer
        self.scorers.append(ctc_scorer)

    def beam_search_score(self, candidate_holder, encoder_outputs):
        """Call the time propagating function, fetch the acoustic score at the current step
        If needed, call the auxiliary scorer and update cand_states in candidate_holder

        Args:
            candidate_holder:  the param cand_seqs and the cand_logits of it is needed
              in the transformer decoder to calculate the output. type: CandidateHolder
            encoder_outputs: the encoder outputs from the transformer encoder.
              type: tuple, (encoder_outputs, input_mask)
        """
        cand_logits = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False
        )
        cand_logits = cand_logits.unstack(
            tf.transpose(candidate_holder.cand_logits, [1, 0, 2])
        )
        cand_seqs = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False
        )
        cand_seqs = cand_seqs.unstack(tf.transpose(candidate_holder.cand_seqs, [1, 0]))
        logits, new_cand_logits, states = self.decoder_one_step(
            cand_logits, cand_seqs, self.states, encoder_outputs
        )
        new_states = candidate_holder.cand_states
        self.states = states
        cand_scores = tf.expand_dims(candidate_holder.cand_scores, axis=1)
        Z = tf.reduce_logsumexp(logits, axis=(1,), keepdims=True)
        logprobs = logits - Z
        new_scores = logprobs + cand_scores  # shape: (cand_num, num_class)
        if self.scorers:
            for scorer in self.scorers:
                other_scores, new_states = scorer.score(candidate_holder, new_scores)
                if other_scores is not None:
                    new_scores += other_scores
        new_cand_logits = tf.transpose(new_cand_logits.stack(), [1, 0, 2])
        return new_scores, new_cand_logits, new_states

    def deal_with_completed(
        self,
        completed_scores,
        completed_seqs,
        completed_length,
        new_scores,
        candidate_holder,
        max_seq_len):
        """Add the new calculated completed seq with its score to completed seqs
        select top beam_size probable completed seqs with these corresponding scores

        Args:
            completed_scores: the scores of completed_seqs
            completed_seqs: historical top beam_size probable completed seqs
            completed_length: the length of completed_seqs
            new_scores: the current time step scores
            candidate_holder:
            max_seq_len: the maximum acceptable output length

        Returns:
            new_completed_scores: new top probable scores
            completed_seqs: new top probable completed seqs
            completed_length: new top probable seq length
        """
        # Add to pool of completed seqs
        new_completed_scores = tf.concat(
            [completed_scores, new_scores[:, self.eos]], axis=0
        )
        cand_seq_len = tf.shape(candidate_holder.cand_seqs)[1]
        eos_tail = tf.fill(
            [tf.shape(candidate_holder.cand_seqs)[0], max_seq_len - cand_seq_len],
            self.eos,
        )
        new_completed_seqs = tf.concat([candidate_holder.cand_seqs, eos_tail], axis=1)
        completed_seqs = tf.concat([completed_seqs, new_completed_seqs], axis=0)
        new_completed_length = tf.fill(
            [tf.shape(new_completed_seqs)[0]], cand_seq_len + 1
        )
        completed_length = tf.concat([completed_length, new_completed_length], axis=0)

        completed_len = tf.shape(new_completed_scores)[0]
        if completed_len > self.beam_size:
            # Rescale scores by sequence length
            completed_length_float = tf.cast(completed_length, tf.float32)
            rescaled_scores = new_completed_scores / completed_length_float
            _, inds = tf.math.top_k(rescaled_scores, k=self.beam_size)
            new_completed_scores = tf.gather(new_completed_scores, inds)
            completed_seqs = tf.gather(completed_seqs, inds)
            completed_length = tf.gather(completed_length, inds)
        return new_completed_scores, completed_seqs, completed_length

    def deal_with_uncompleted(
        self,
        new_scores,
        new_cand_logits,
        new_states,
        candidate_holder):
        """select top probable candidate seqs from new predictions with its scores
        update candidate_holder based on top probable candidates

        Args:
            new_scores: the current time step prediction scores
            new_cand_logits: historical prediction scores
            new_states: updated states
            candidate_holder:

        Returns:
            candidate_holder: cand_seqs, cand_logits, cand_states,
              cand_scores, cand_parents will be updated here and sent
              to next time step
        """
        cand_seqs = candidate_holder.cand_seqs
        num_cands = tf.shape(candidate_holder.cand_seqs)[0]

        # Deal with non-completed candidates
        not_eos = tf.range(self.num_class) != self.eos
        not_eos = tf.range(self.num_class)[not_eos]
        parents, syms = tf.meshgrid(
            tf.range(num_cands), tf.range(self.num_class), indexing="ij"
        )
        new_scores = tf.gather(new_scores, not_eos, axis=1)
        parents = tf.gather(parents, not_eos, axis=1)
        syms = tf.gather(syms, not_eos, axis=1)
        new_scores_flat = tf.reshape(new_scores, [-1])
        parents_flat = tf.reshape(parents, [-1])
        syms_flat = tf.reshape(syms, [-1])

        new_scores_flat_len = tf.shape(new_scores_flat)[0]
        if new_scores_flat_len > self.beam_size:
            _, inds = tf.math.top_k(new_scores_flat, k=self.beam_size)
        else:
            inds = tf.range(new_scores_flat_len)
        cand_syms = tf.gather(syms_flat, inds)
        cand_parents = tf.gather(parents_flat, inds)
        # Update beam state
        cand_scores = tf.gather(new_scores_flat, inds)
        cand_logits = tf.gather(new_cand_logits, cand_parents)
        cand_states = [tf.gather(state, cand_parents) for state in new_states]
        cand_seqs = tf.gather(cand_seqs, cand_parents)
        cand_syms = tf.expand_dims(cand_syms, 1)
        cand_seqs = tf.concat([cand_seqs, cand_syms], axis=1)
        cand_states[0] = cand_seqs
        candidate_holder = CandidateHolder(
            cand_seqs, cand_logits, cand_states, cand_scores, cand_parents
        )
        return candidate_holder

    def __call__(self, cand_seqs, cand_states, init_states, encoder_outputs):
        """
        Args:
            cand_seqs: TensorArray list, element shape: [beam]
            cand_states: [history_predictions]
            init_states: state list
            encoder_outputs: (encoder_outputs, memory_mask, ...)

        Returns:
            completed_seqs: the sequence with highest score
        """
        cand_logits = tf.fill([tf.shape(cand_seqs)[0], 0, self.num_class], 0.0)
        cand_scores = tf.fill([tf.shape(cand_seqs)[0]], 0.0)
        cand_scores = tf.cast(cand_scores, tf.float32)
        cand_parents = tf.fill([1], 0)
        candidate_holder = CandidateHolder(
            cand_seqs, cand_logits, cand_states, cand_scores, cand_parents
        )
        self.states = init_states
        if self.lm_model is not None:
            self.lm_model.reset()

        max_seq_len = encoder_outputs[0].shape[1]
        completed_seqs = tf.fill([0, max_seq_len], self.eos)
        completed_length = tf.fill([0], 1)
        completed_scores = tf.fill([0], 0.0)
        for pos in tf.range(max_seq_len):
            # compute new scores
            new_scores, new_cand_logits, new_states = self.beam_search_score(
                candidate_holder, encoder_outputs
            )
            # extract seqs with end symbol
            (
                completed_scores,
                completed_seqs,
                completed_length,
            ) = self.deal_with_completed(
                completed_scores,
                completed_seqs,
                completed_length,
                new_scores,
                candidate_holder,
                max_seq_len,
            )
            if (pos + 1) >= max_seq_len:
                break
            not_eos = tf.range(self.num_class) != self.eos
            not_eos = tf.range(self.num_class)[not_eos]
            new_scores = tf.gather(new_scores, not_eos, axis=1)
            # Terminate if all candidates are already worse than all completed seqs
            min_completed_score = tf.reduce_min(completed_scores)
            max_new_score = tf.reduce_max(new_scores)
            cand_seq_len = tf.shape(candidate_holder.cand_seqs)[1] + 1
            cand_seq_len_float = tf.cast(cand_seq_len, tf.float32)
            new_scores_rescale = new_scores / cand_seq_len_float
            max_new_score_rescale = tf.reduce_max(new_scores_rescale)
            completed_length_float = tf.cast(completed_length, tf.float32)
            rescale_scores = completed_scores / completed_length_float
            min_completed_score_rescale = tf.reduce_min(rescale_scores)
            # Strict the statement of termination by scores and normalized scores(by length)
            if (
                max_new_score < min_completed_score
                and max_new_score_rescale < min_completed_score_rescale
            ):
                break
            candidate_holder = self.deal_with_uncompleted(
                new_scores, new_cand_logits, new_states, candidate_holder
            )
            encoder_output, input_mask = encoder_outputs
            encoder_output = tf.tile(
                encoder_output[:1], [tf.shape(candidate_holder.cand_seqs)[0], 1, 1]
            )
            input_mask = tf.tile(
                input_mask[:1], [tf.shape(candidate_holder.cand_seqs)[0], 1, 1, 1]
            )
            encoder_outputs = (encoder_output, input_mask)

        # Sort completed seqs
        completed_length_float = tf.cast(completed_length, tf.float32)
        rescaled_scores = completed_scores / completed_length_float
        _, inds = tf.math.top_k(rescaled_scores, k=1)
        length = completed_length[inds[0]]
        return tf.cast(
            tf.expand_dims(completed_seqs[inds[0]][1:length], axis=0), tf.int64
        )
