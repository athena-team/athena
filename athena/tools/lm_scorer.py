import numpy as np
import kenlm


class NGramScorer(object):
    """
    KenLM language model
    """

    def __init__(self, lm_path, sos, eos, num_syms, lm_weight=0.1):
        """
        Basic params will be initialized, the kenlm model will be created from
        the lm_path
        Args:
            lm_path: the saved lm model path
            sos: start symbol
            eos: end symbol
            num_syms: number of classes
            lm_weight: the lm weight
        """
        self.lang_model = kenlm.Model(lm_path)
        self.state_index = 0
        self.sos = sos
        self.eos = eos
        self.num_syms = num_syms
        self.lm_weight = lm_weight
        kenlm_state = kenlm.State()
        self.lang_model.BeginSentenceWrite(kenlm_state)
        self.cand_kenlm_states = np.array([[kenlm_state] * num_syms])

    def score(self, candidate_holder, new_scores):
        """
        Call this function to compute the NGram score of the next prediction
        based on historical predictions, the scoring function shares a common interface
        Args:
            candidate_holder:
        Returns:
            score: the NGram weighted score
            cand_states:
        """
        cand_seqs = candidate_holder.cand_seqs
        cand_parents = candidate_holder.cand_parents
        cand_syms = cand_seqs[:, -1]
        score = self.get_score(cand_parents, cand_syms, self.lang_model)
        score = self.lm_weight * score
        return score, candidate_holder.cand_states

    def get_score(self, cand_parents, cand_syms, lang_model):
        """
        the saved lm model will be called here
        Args:
            cand_parents: last selected top candidates
            cand_syms: last selected top char index
            lang_model: the language model
        Return:
            scores: the lm scores
        """
        scale = 1.0 / np.log10(np.e)  # convert log10 to ln

        num_cands = len(cand_syms)
        scores = np.zeros((num_cands, self.num_syms))
        new_states = np.zeros((num_cands, self.num_syms), dtype=object)
        chars = [str(x) for x in range(self.num_syms)]
        chars[self.sos] = "<s>"
        chars[self.eos] = "</s>"
        chars[0] = "<space>"

        for i in range(num_cands):
            parent = cand_parents[i]
            kenlm_state_list = self.cand_kenlm_states[parent]
            kenlm_state = kenlm_state_list[cand_syms[i]]
            for sym in range(self.num_syms):
                char = chars[sym]
                out_state = kenlm.State()
                score = scale * lang_model.BaseScore(kenlm_state, char, out_state)
                scores[i, sym] = score
                new_states[i, sym] = out_state
        self.cand_kenlm_states = new_states
        return scores
