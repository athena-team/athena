"""WFST decoder for Seq2Seq model
Python version decoder which treat Seq2Seq models(attention, transformer, CTC)
as acoustic model. Record inner states of Seq2Seq models in token and execute
token passing algorithm in WFST decoding graph.
"""
import sys
import numpy as np
import openfst_python as fst
import tensorflow as tf
from absl import logging
from .ctc_scorer import CTCPrefixScorer
from collections import namedtuple

CandidateHolder = namedtuple(
    "CandidateHolder",
    ["cand_seqs", "cand_states"],
)

class LatticeArc:
    """Arc used in Token
    acoustic cost and graph cost are stored seperately.
    """
    def __init__(self, ilabel, olabel, weight, nextstate):
        self.ilabel = ilabel
        self.olabel = olabel
        self.weight = weight
        self.nextstate = nextstate

class Token:
    """Token used in token passing decode algorithm.
    The token can be linked by another token or None.
    The token record the linked arc, cost and inner packed states
    used in model
    """
    def __init__(self, arc, acoustic_cost, prev_tok=None, cur_label=None, inner_packed_states=None, cand_states=None):
        self.prev_tok = prev_tok
        self.cur_label = cur_label
        self.inner_packed_states = inner_packed_states
        self.cand_states = cand_states
        self.arc = LatticeArc(arc.ilabel, arc.olabel,
                (float(arc.weight), acoustic_cost), arc.nextstate)
        if prev_tok is not None:
            self.cost = prev_tok.cost + float(arc.weight) + acoustic_cost
        else:
            self.cost = float(arc.weight) + acoustic_cost
        self.rescaled_cost = -1.0


class WFSTDecoder:

    def __init__(self, hparams, num_class, sos, eos, time_propagate):
        self.cur_toks = {}
        self.prev_toks = {}
        self.completed_token_pool = []
        self.num_steps_decoded = -1
        self.beam_delta = 0.5
        self.fst = fst.Fst.read(hparams.wfst_path)
        logging.info("read WFST graph successfully")
        self.acoustic_scale = hparams.acoustic_scale
        self.max_active = hparams.max_active
        self.min_active = hparams.min_active
        self.beam = hparams.wfst_beam
        self.sos = sos
        self.eos = eos
        self.num_syms = num_class
        self.max_seq_len = None
        self.inference_one_step_fn = time_propagate
        self.step = None
        self.ctc_scorer = None
        self.scorers = []

        self.vocab = {}
        with open(hparams.vocab,'r') as f:
            for line in f:
                char = line.strip().split()[0]
                idx = line.strip().split()[1]
                self.vocab[char] = int(idx)
        self.words = []
        with open(hparams.osymbols,'r') as f:
            for line in f:
                word = line.strip().split()[0]
                self.words.append(word)

    """Decode the input samples using seq2seq models and WFST graph"""
    @staticmethod
    def build_decoder(hparams, num_class, sos, eos, time_propagate, lm_model=None):
        """Init decoder set some inner variance.
        Args:
            hparams: hyper param
            num_class: number of classes
            sos: start symbol
            eos: end symbol
            time_propagate: callback function of seq2seq model. the function
                       take encoder_outputs,input label and inner states,
                       then produce logscores and inner states.
                       the function's signature is: logscores, packed_inner_states =
                       fn(encoder_outputs, label_input, packed_inner_states)
            auxiliary_model: other auxiliary models
        """
        wfst_decoder = WFSTDecoder(hparams, num_class, sos, eos, time_propagate)
        if hparams.ctc_weight != 0:
            ctc_scorer = CTCPrefixScorer(
                eos,
                ctc_beam=hparams.wfst_beam*2,
                num_classes=num_class,
                ctc_weight=hparams.ctc_weight,
            )
            wfst_decoder.set_ctc_scorer(ctc_scorer)
        return wfst_decoder

    def set_ctc_scorer(self, ctc_scorer):
        """ set the ctc_scorer
        Args:
            ctc_scorer: the ctc scorer
        """
        self.ctc_scorer = ctc_scorer
        self.scorers.append(ctc_scorer)

    def __call__(self, history_predictions, init_cand_states, step, encoder_outputs):
        """using seq2seq model and WFST graph to decode input utterance
        Args:
            encoder_outputs: outputs of encoder for encoder-decoder structure model.
                             or just CTC outputs for ctc model.
            step: time step
            history_predictions: initial historical predictions
            init_cand_states: initial packed states for inference_one_step_fn callback function
        """
        self.step = step
        self.max_seq_len = encoder_outputs[0].shape[1]
        init_cand_logits = tf.fill([0, self.num_syms], 0.0)
        cand_states = []
        for states in init_cand_states:
            cand_states.append(states[0])
        self.init_decoding(init_cand_logits, cand_states)
        while not self.end_detect():
            encoder_output, input_mask = encoder_outputs
            encoder_output = tf.tile(
                encoder_output[:1], [len(self.cur_toks), 1, 1]
            )
            input_mask = tf.tile(
                input_mask[:1], [len(self.cur_toks), 1, 1, 1]
            )
            encoder_outputs = (encoder_output, input_mask)
            self.prev_toks = self.cur_toks
            self.cur_toks = {}
            weight_cutoff = self.process_emitting(encoder_outputs)
            self.process_nonemitting(weight_cutoff)
        best_path = self.get_best_path()
        best_path = tf.convert_to_tensor(np.array(best_path))
        return tf.cast(
            tf.expand_dims(best_path, axis=0), tf.int64
        )

    def end_detect(self):
        """determine whether to stop propagating"""
        if self.cur_toks and self.num_steps_decoded < self.max_seq_len:
            return False
        else:
            return True

    def init_decoding(self, initial_packed_states, cand_states):
        """Init decoding states for every input utterance
        Args:
            initial_packed_states: initial packed states for callback function
        """
        self.cur_toks = {}
        self.prev_toks = {}
        self.completed_token_pool = []
        start_state = self.fst.start()
        assert start_state != -1
        dummy_arc = LatticeArc(0, 0, 0.0, start_state)
        self.cur_toks[start_state] = Token(dummy_arc, 0.0, None, [self.sos], initial_packed_states, cand_states)
        self.num_steps_decoded = 0
        self.process_nonemitting(float('inf'))

    def deal_completed_token(self, state, eos_score):
        """deal completed token and rescale scores
        Args:
            state: state of the completed token
            eos_score: acoustic score of eos
        """
        tok = self.prev_toks[state]
        if self.fst.final(state).to_string() != fst.Weight.Zero('tropical').to_string():
            tok.rescaled_cost = ((tok.cost + (-eos_score) +
                float(self.fst.final(state)))/self.num_steps_decoded)
            self.completed_token_pool.append(tok)
        queue = []
        tmp_toks = {}
        queue.append(state)
        tmp_toks[state] = tok
        while queue:
            state = queue.pop()
            tok = tmp_toks[state]
            for arc in self.fst.arcs(state):
                if arc.ilabel == 0:
                    new_tok = Token(arc, 0.0, tok, tok.cur_label, tok.inner_packed_states, tok.cand_states)
                    if self.fst.final(arc.nextstate).to_string() != fst.Weight.Zero('tropical').to_string():
                        new_tok.rescaled_cost = ((new_tok.cost +
                            float(self.fst.final(arc.nextstate)))/self.num_steps_decoded)
                        self.completed_token_pool.append(new_tok)
                    else:
                        tmp_toks[arc.nextstate] = new_tok
                        queue.append(arc.nextstate)

    def process_emitting(self, encoder_outputs):
        """Process one step emitting states using callback function
        Args:
            encoder_outputs: encoder outputs
        Returns:
            next_weight_cutoff: cutoff for next step
        """
        state2id = {}
        history_logits_array = []
        history_logits = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False
        )
        cand_seqs_ori_array = []
        cand_seqs = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, clear_after_read=False
        )
        num_states = len(list(self.prev_toks.values())[0].cand_states)
        cand_states_array = [[] for i in range(num_states)] # num of states in cand_states
        cand_states = []
        for idx, state in enumerate(self.prev_toks):
            state2id[state] = idx
            cand_seqs_ori_array.append(self.prev_toks[state].cur_label)
            history_logits_array.append(self.prev_toks[state].inner_packed_states)
            for i, cand_state in enumerate(self.prev_toks[state].cand_states):
                cand_states_array[i].append(cand_state)
        for cand_state in cand_states_array:
            cand_states.append(tf.convert_to_tensor(cand_state)) # [(beam, hsize)*num_states]
        history_logits_array = np.transpose(np.array(history_logits_array), [1, 0, 2]) # [time, beam, S]
        history_logits = history_logits.unstack(tf.convert_to_tensor(history_logits_array))
        cand_seqs_array = tf.convert_to_tensor(np.array(cand_seqs_ori_array), dtype=tf.int32)
        cand_seqs = cand_seqs.unstack(tf.transpose(cand_seqs_array, [1, 0]))
        all_log_scores, new_cand_logits, step = self.inference_one_step_fn(history_logits,
                                                                           cand_seqs,
                                                                           self.step,
                                                                           encoder_outputs)
        candidate_holder = CandidateHolder(cand_seqs_array, cand_states)
        Z = tf.reduce_logsumexp(all_log_scores, axis=(1,), keepdims=True)
        all_log_scores = all_log_scores - Z
        if self.scorers:
            for scorer in self.scorers:
                other_scores, new_cand_states = scorer.score(candidate_holder, all_log_scores)
                if other_scores is not None:
                    all_log_scores += other_scores
        self.step = step
        new_cand_logits = tf.transpose(new_cand_logits.stack(), [1, 0, 2]).numpy() # [beam, time, S]

        weight_cutoff, adaptive_beam, best_token = self.get_cutoff()
        next_weight_cutoff = float('inf')
        if best_token is not None:
            for arc in self.fst.arcs(best_token.arc.nextstate):
                if arc.ilabel != 0:
                    ac_cost = (-all_log_scores[state2id[best_token.arc.nextstate]][arc.ilabel-1] *
                            self.acoustic_scale)
                    new_weight = float(arc.weight) + best_token.cost + ac_cost
                    if new_weight + adaptive_beam < next_weight_cutoff:
                        next_weight_cutoff = new_weight + adaptive_beam
        for state, tok in self.prev_toks.items():
            seq_id = state2id[state]
            if tok.cost <= weight_cutoff:
                if self.eos == np.argmax(all_log_scores[seq_id]):
                    self.deal_completed_token(state, all_log_scores[seq_id][self.eos])
                    continue
                for arc in self.fst.arcs(state):
                    if arc.ilabel != 0:
                        ac_cost = -all_log_scores[seq_id][arc.ilabel-1] * self.acoustic_scale
                        new_weight = float(arc.weight) + tok.cost + ac_cost
                        if new_weight < next_weight_cutoff:
                            inner_packed_state = new_cand_logits[seq_id]
                            one_cand_state = [s[seq_id] for s in new_cand_states]
                            new_tok = Token(arc, ac_cost, tok, tok.cur_label+[arc.ilabel-1],
                                    inner_packed_state, one_cand_state)
                            if new_weight + adaptive_beam < next_weight_cutoff:
                                next_weight_cutoff = new_weight + adaptive_beam
                            if arc.nextstate in self.cur_toks:
                                if self.cur_toks[arc.nextstate].cost > new_tok.cost:
                                    self.cur_toks[arc.nextstate] = new_tok
                            else:
                                self.cur_toks[arc.nextstate] = new_tok
        self.num_steps_decoded += 1
        return next_weight_cutoff

    def process_nonemitting(self, cutoff):
        """Process one step nonemitting states
        """
        queue = []
        for state, _ in self.cur_toks.items():
            queue.append(state)
        while queue:
            state = queue.pop()
            tok = self.cur_toks[state]
            if tok.cost > cutoff:
                continue
            for arc in self.fst.arcs(state):
                if arc.ilabel == 0:
                    new_tok = Token(arc, 0.0, tok, tok.cur_label, tok.inner_packed_states, tok.cand_states)
                    if new_tok.cost < cutoff:
                        if arc.nextstate in self.cur_toks:
                            if self.cur_toks[arc.nextstate].cost > new_tok.cost:
                                self.cur_toks[arc.nextstate] = new_tok
                                queue.append(arc.nextstate)
                        else:
                            self.cur_toks[arc.nextstate] = new_tok
                            queue.append(arc.nextstate)

    def get_cutoff(self):
        """get cutoff used in current and next step
        Returns:
            beam_cutoff: beam cutoff
            adaptive_beam: adaptive beam
            best_token: best token this step
        """
        best_cost = float('inf')
        best_token = None
        if(self.max_active == sys.maxsize
                and self.min_active == 0):
            for _, tok in self.prev_toks.items():
                if tok.cost < best_cost:
                    best_cost = tok.cost
                    best_token = tok
                adaptive_beam = self.beam
                beam_cutoff = best_cost + self.beam
                return beam_cutoff, adaptive_beam, best_token
        else:
            tmp_array = []
            for _, tok in self.prev_toks.items():
                tmp_array.append(tok.cost)
                if tok.cost < best_cost:
                    best_cost = tok.cost
                    best_token = tok
            beam_cutoff = best_cost + self.beam
            min_active_cutoff = float('inf')
            max_active_cutoff = float('inf')
            if len(tmp_array) > self.max_active:
                np_tmp_array = np.array(tmp_array)
                k = self.max_active
                max_active_cutoff = np_tmp_array[np.argpartition(np_tmp_array, k-1)[k-1]]
            if max_active_cutoff < beam_cutoff:
                adaptive_beam = max_active_cutoff - best_cost + self.beam_delta
                return max_active_cutoff, adaptive_beam, best_token
            if len(tmp_array) > self.min_active:
                np_tmp_array = np.array(tmp_array)
                k = self.min_active
                if k == 0:
                    min_active_cutoff = best_cost
                else:
                    min_active_cutoff = np_tmp_array[np.argpartition(np_tmp_array, k-1)[k-1]]
            if min_active_cutoff > beam_cutoff:
                adaptive_beam = min_active_cutoff - best_cost + self.beam_delta
                return min_active_cutoff, adaptive_beam, best_token
            else:
                adaptive_beam = self.beam
                return beam_cutoff, adaptive_beam, best_token

    def get_best_path(self):
        """get decoding result in best completed path
        Returns:
            ans: id array of decoding results
        """
        ans = []
        if not self.completed_token_pool:
            return ans
        best_completed_tok = self.completed_token_pool[0]
        for tok in self.completed_token_pool:
            if best_completed_tok.rescaled_cost > tok.rescaled_cost:
                best_completed_tok = tok
        arcs_reverse = []
        tok = best_completed_tok
        while tok:
            arcs_reverse.append(tok.arc)
            tok = tok.prev_tok
        assert arcs_reverse[-1].nextstate == self.fst.start()
        arcs_reverse.pop()
        for arc in arcs_reverse[::-1]:
            if arc.olabel != 0:
                ans.append(arc.olabel)
        ans = ''.join([self.words[int(idx)] for idx in ans])
        ans = [self.vocab[prediction] for prediction in ans]
        return ans
