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



def tf_index_select(input_, dim, indices):
    """
    input_(tensor): input tensor
    dim(int): dimension
    indices(list): selected indices list
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape) - 1
    shape[dim] = 1

    tmp = []
    for idx in indices:
        begin = [0] * len(shape)
        begin[dim] = idx
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)

    return res


# class CTCPrefixScorer(object):
#     """Decoder interface wrapper for CTCPrefixScore."""
#
#     def __init__(self, ctc: torch.nn.Module, eos: int):
#         """Initialize class.
#
#         Args:
#             ctc (torch.nn.Module): The CTC implementaiton.
#                 For example, :class:`espnet.nets.pytorch_backend.ctc.CTC`
#             eos (int): The end-of-sequence id.
#
#         """
#         self.ctc = ctc
#         self.eos = eos
#         self.impl = None
#
#
#     def select_state(self, state, i, new_id=None):
#         """Select state with relative ids in the main beam search.
#
#         Args:
#             state: Decoder state for prefix tokens
#             i (int): Index to select a state in the main beam search
#             new_id (int): New label id to select a state if necessary
#
#         Returns:
#             state: pruned state
#
#         """
#         if type(state) == tuple:
#             if len(state) == 2:  # for CTCPrefixScore
#                 sc, st = state
#                 return sc[i], st[i]
#             else:  # for CTCPrefixScoreTH (need new_id > 0)
#                 r, log_psi, f_min, f_max, scoring_idmap = state
#                 s = log_psi[i, new_id].expand(log_psi.size(1))
#                 if scoring_idmap is not None:
#                     return r[:, :, i, scoring_idmap[i, new_id]], s, f_min, f_max
#                 else:
#                     return r[:, :, i, new_id], s, f_min, f_max
#         return None if state is None else state[i]
#
#     def score_partial(self, y, ids, state, x):
#         """Score new token.
#
#         Args:
#             y (torch.Tensor): 1D prefix token
#             next_tokens (torch.Tensor): torch.int64 next token to score
#             state: decoder state for prefix tokens
#             x (torch.Tensor): 2D encoder feature that generates ys
#
#         Returns:
#             tuple[torch.Tensor, Any]:
#                 Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
#                 and next state for ys
#
#         """
#         prev_score, state = state
#         presub_score, new_st = self.impl(y.cpu(), ids.cpu(), state)
#         tscore = torch.as_tensor(
#             presub_score - prev_score, device=x.device, dtype=x.dtype
#         )
#         return tscore, (presub_score, new_st)
#
#
#     def batch_score_partial(self, y, ids, state, x):
#         """Score new token.
#
#         Args:
#             y (torch.Tensor): 1D prefix token
#             ids (torch.Tensor): torch.int64 next token to score
#             state: decoder state for prefix tokens
#             x (torch.Tensor): 2D encoder feature that generates ys
#
#         Returns:
#             tuple[torch.Tensor, Any]:
#                 Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
#                 and next state for ys
#
#         """
#         batch_state = (
#             (
#                 torch.stack([s[0] for s in state], dim=2),
#                 torch.stack([s[1] for s in state]),
#                 state[0][2],
#                 state[0][3],
#             )
#             if state[0] is not None
#             else None
#         )
#         return self.impl(y, batch_state, ids)
#
#     def extend_prob(self, x: torch.Tensor):
#         """Extend probs for decoding.
#
#         This extention is for streaming decoding
#         as in Eq (14) in https://arxiv.org/abs/2006.14941
#
#         Args:
#             x (torch.Tensor): The encoded feature tensor
#
#         """
#         logp = self.ctc.log_softmax(x.unsqueeze(0))
#         self.impl.extend_prob(logp)
#
#     def extend_state(self, state):
#         """Extend state for decoding.
#
#         This extention is for streaming decoding
#         as in Eq (14) in https://arxiv.org/abs/2006.14941
#
#         Args:
#             state: The states of hyps
#
#         Returns: exteded state
#
#         """
#         new_state = []
#         for s in state:
#             new_state.append(self.impl.extend_state(s))
#
#         return new_state

class CTCPrefixScoreTH(object):
    """Batch processing of CTCPrefixScore

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the label probablities for multiple
    hypotheses simultaneously
    See also Seki et al. "Vectorized Beam Search for CTC-Attention-Based
    Speech Recognition," In INTERSPEECH (pp. 3825-3829), 2019.
    """

    def __init__(self, x, xlens, blank, eos, margin=0):
        """Construct CTC prefix scorer

        :param torch.Tensor x: input label posterior sequences (B, T, O) B: batch*beam -> batch; O:V_class
        :param torch.Tensor xlens: input lengths (B,)
        :param int blank: blank label id
        :param int eos: end-of-sequence id
        :param int margin: margin parameter for windowing (0 means no windowing)
        """
        # In the comment lines,
        # we assume T: input_length, B: batch size, W: beam width, O: output dim.

        # x = tf.expand_dims(x[0], 0)
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.batch = x.shape[0]  # x.shape[0] or 1
        self.input_length = x.shape[1]
        self.odim = x.shape[2]
        self.dtype = x.dtype
        # Pad the rest of posteriors in the batch
        # TODO(takaaki-hori): need a better way without for-loops
        for i, l in enumerate(xlens):
            if l < self.input_length:
                x[i, l:, :] = self.logzero
                x[i, l:, blank] = 0
        # Reshape input x
        xn = tf.transpose(x, perm=[1, 0, 2])  # (B, T, O) -> (T, B, O) => (B*W, T, O) -> (T, B*W, O)
        xb = tf.tile(tf.expand_dims(xn[:, :, self.blank], 2), multiples=[1, 1, self.odim])
        self.x = tf.stack([xn, xb])  # (2, T, B, O)
        self.end_frames = xlens - 1  # (B)

        # Setup CTC windowing
        self.margin = margin
        if margin > 0:
            self.frame_ids = tf.range(
                self.input_length, dtype=self.dtype
            )
        # Base indices for index conversion
        self.idx_bh = None
        self.idx_b = tf.range(self.batch)
        self.idx_bo = tf.expand_dims(self.idx_b * self.odim, 1)

    def __call__(self, y, state, scoring_ids=None, att_w=None):
        """Compute CTC prefix scores for next labels

        :param y: tensor(shape=[W, L]),
                    prefix label sequences
        :param tuple state: previous CTC state
            tuple(
                tensor(shape=[T , 2, W]),
                tensor(shape=[W, O]),
                0, 0
              )
        :param torch.Tensor scoring_ids: scores for pre-selection of hypotheses [Beam, Beam * pre_beam_ratio]
        :param torch.Tensor att_w: attention weights to decide CTC window
        :return new_state, ctc_local_scores (BW, O)
        """
        if state is None:
            y = tf.reshape(y[0], [1, -1])
            scoring_ids = tf.reshape(scoring_ids[0], [1, -1])
        output_length = len(y[0]) - 1  # ignore sos
        last_ids = [yi[-1] for yi in y]  # last output label ids
        n_bh = len(last_ids)  # batch * hyps
        n_hyps = n_bh // self.batch  # assuming each utterance has the same # of hyps
        self.scoring_num = scoring_ids.shape[-1] if scoring_ids is not None else 0
        # prepare state info
        if state is None:

            # r_prev1 = tf.fill(
            #     [self.input_length, 2, self.batch, n_hyps], self.logzero
            # )  # (46, 2, 1, 20)
            # r_prev1[:, 1] = tf.expand_dims(tf.cumsum(self.x[0, :, :, self.blank], 0), 2)  # (46, 20, 1)

            r_prev = tf.fill([self.input_length, self.batch, n_hyps], self.logzero)
            r_prev = tf.stack([r_prev, tf.expand_dims(tf.cumsum(self.x[0, :, :, self.blank], 0), 2)], axis=1)
            r_prev = tf.reshape(r_prev, [-1, 2, n_bh])
            r_prev = r_prev.numpy()
            s_prev = 0.0
            # print("r_prev1.shape:", r_prev.shape)  # (46, 2, 1)
            # print("s_prev1:", s_prev)  # 0.0
            f_min_prev = 0
            f_max_prev = 1
        else:
            r_prev, s_prev, f_min_prev, f_max_prev = state
            # print("r_prev.shape:", r_prev.shape)  # (46, 2, 1, 30) ?-> (46, 2, 1) corr:( T, 2, n_hyps)
            # print("s_prev.shape:", s_prev.shape)  # (1, 4233)  corr: (n_hyps, V_class)

        # select input dimensions for scoring
        if self.scoring_num > 0:
            scoring_idmap = -np.ones([n_bh, self.odim])
            snum = self.scoring_num
            if self.idx_bh is None or n_bh > len(self.idx_bh):
                self.idx_bh = tf.reshape(tf.range(n_bh), [-1, 1])
            scoring_idmap[self.idx_bh.numpy()[:n_bh], scoring_ids.numpy()] = np.arange(snum)
            scoring_idx = tf.reshape(
                scoring_ids + tf.reshape(np.tile(self.idx_bo.numpy(), (1, n_hyps)), [-1, 1]), -1
            )
            x_ = tf.reshape(tf.expand_dims(tf.gather(
                tf.reshape(self.x, [2, -1, self.batch * self.odim]), scoring_idx, axis=2
            ), 2), [2, -1, n_bh, snum])
        else:
            scoring_ids = None
            scoring_idmap = None
            snum = self.odim
            x_ = tf.reshape(tf.tile(tf.expand_dims(self.x, 3), [1, 1, 1, n_hyps, 1]), [2, -1, n_bh, snum])

        # new CTC forward probs are prepared as a (T x 2 x BW x S) tensor
        # that corresponds to r_t^n(h) and r_t^b(h) in a batch.
        r = tf.fill(
            [self.input_length, 2, n_bh, snum],
            self.logzero,
        )
        r = r.numpy()
        if output_length == 0:
            r[0, 0] = x_[0, 0].numpy()
        r_sum = tf.reduce_logsumexp(r_prev, 1)
        log_phi = tf.tile(tf.expand_dims(r_sum, 2).numpy(), [1, 1, snum])
        log_phi = log_phi.numpy()
        if scoring_ids is not None:
            for idx in range(n_bh):
                pos = scoring_idmap[idx, last_ids[idx]]
                if pos >= 0:
                    log_phi[:, idx, int(pos)] = r_prev[:, 1, idx]
        else:
            for idx in range(n_bh):
                log_phi[:, idx, last_ids[idx].numpy()] = r_prev[:, 1, idx]

        # decide start and end frames based on attention weights
        if att_w is not None and self.margin > 0:
            f_arg = tf.matmul(att_w, self.frame_ids).numpy()
            f_min = max(int(f_arg.min()), f_min_prev)
            f_max = max(int(f_arg.max()), f_max_prev)
            start = min(f_max_prev, max(f_min - self.margin, output_length, 1))
            end = min(f_max + self.margin, self.input_length)
        else:
            f_min = f_max = 0
            start = max(output_length, 1)
            end = self.input_length

        # compute forward probabilities log(r_t^n(h)) and log(r_t^b(h))
        for t in range(start, end):
            rp = r[t - 1]  # shape:(2, 1, 30)
            rr = tf.reshape(tf.stack([rp[0], log_phi[t - 1], rp[0], rp[1]]), [
                # rp[0]:(1,30), log_phi[t-1]:(30,30) log_phi: (T, pre_beam, pre_beam) -> (T, hpys, pre_beam)
                2, 2, n_bh, snum
            ])
            r[t] = tf.reduce_logsumexp(rr, 1) + x_[:, t]

        # compute log prefix probabilites log(psi)
        log_phi_x = tf.concat([tf.expand_dims(log_phi[0], 0), log_phi[:-1]], axis=0) + x_[0]
        if scoring_ids is not None:
            log_psi = tf.fill(
                [n_bh, self.odim], self.logzero
            )
            log_psi_ = tf.reduce_logsumexp(
                tf.concat([log_phi_x[start:end], tf.expand_dims(r[start - 1, 0], 0)], axis=0),
                axis=0
            )
            log_psi = log_psi.numpy()  # [20, 4233]
            for si in range(n_bh):
                log_psi[si, scoring_ids[si].numpy()] = log_psi_[si].numpy()
        else:
            log_psi = tf.reduce_logsumexp(
                tf.concat([log_phi_x[start:end], tf.expand_dims(r[start - 1, 0], 0)], axis=0),
                axis=0,
            )

        for si in range(n_bh):  # n_bh=1 or 20
            log_psi[si, self.eos] = r_sum[self.end_frames[si // n_hyps], si]  # self.end_frames=[45] err:(si//n_hyps)==1

        # exclude blank probs
        log_psi[:, self.blank] = self.logzero

        return (log_psi - s_prev), (r, log_psi, f_min, f_max, scoring_idmap)

    def index_select_state(self, state, best_ids):
        """Select CTC states according to best ids

        :param state    : CTC state
        :param best_ids : index numbers selected by beam pruning (B, W)
        :return selected_state
        """
        r, s, f_min, f_max, scoring_idmap = state
        # convert ids to BHO space
        n_bh = len(s)
        n_hyps = n_bh // self.batch
        vidx = tf.reshape(best_ids + tf.reshape(self.idx_b * (n_hyps * self.odim), [-1, 1]), [-1])
        # select hypothesis scores
        s_new = tf_index_select(tf.reshape(s, [-1]), 0, vidx)  # shape:(400)
        s_new = tf.reshape(tf.tile(tf.reshape(s_new, [-1, 1]), [1, self.odim]), [n_bh, self.odim])  # reshape should 84660, not 1693200
        # convert ids to BHS space (S: scoring_num)
        if scoring_idmap is not None:
            snum = self.scoring_num
            hyp_idx = tf.reshape(best_ids // self.odim + tf.reshape(self.idx_b * n_hyps, [-1, 1]),
                                 [-1]
                                 )
            label_ids = tf.reshape(np.fmod(best_ids.numpy(), self.odim), [-1])
            score_idx = scoring_idmap[hyp_idx, label_ids]
            score_idx[score_idx == -1] = 0
            vidx = score_idx + hyp_idx * snum
        else:
            snum = self.odim
        # select forward probabilities
        r_new = tf.reshape(tf_index_select(tf.reshape(r, [-1, 2, n_bh * snum]), 2, vidx),
                           [-1, 2, n_bh]
                           )
        return r_new, s_new, f_min, f_max
