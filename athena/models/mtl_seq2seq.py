# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Wubo Li
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
# Only support eager mode and TF>=2.0.0
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes
""" a implementation of deep speech 2 model can be used as a sample for ctc model """

import tensorflow as tf
from tensorflow.keras.layers import Dense
from .base import BaseModel
from ..loss import CTCLoss
from ..metrics import CTCAccuracy
from .speech_transformer import SpeechTransformer, SpeechTransformer2
from ..utils.hparam import register_and_parse_hparams
from ..tools.beam_search import BeamSearchDecoder
from ..tools.ctc_scorer import CTCPrefixScorer
from ..tools.lm_scorer import NGramScorer


class MtlTransformerCtc(BaseModel):
    """ In speech recognition, adding CTC loss to Attention-based seq-to-seq model is known to
    help convergence. It usually gives better results than using attention alone.
    """

    SUPPORTED_MODEL = {
        "speech_transformer": SpeechTransformer,
        "speech_transformer2": SpeechTransformer2,
    }

    default_config = {
        "model": "speech_transformer",
        "model_config":  {"return_encoder_output": True},
        "mtl_weight": 0.5
    }

    def __init__(self, num_classes, sample_shape, config=None):
        super().__init__()
        self.num_classes = num_classes + 1
        self.sos = self.num_classes - 1
        self.eos = self.num_classes - 1

        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        self.loss_function = CTCLoss(blank_index=-1)
        self.metric = CTCAccuracy()
        self.model = self.SUPPORTED_MODEL[self.hparams.model](
            num_classes, sample_shape, self.hparams.model_config
        )
        self.decoder = Dense(self.num_classes)
        self.ctc_logits = None

    def call(self, samples, training=None):
        """ call function in keras layers """
        output, encoder_output = self.model(samples, training=training)
        self.ctc_logits = self.decoder(encoder_output, training=training)
        return output

    def get_loss(self, logits, samples, training=None):
        """ get loss used for training """
        logit_length = self.compute_logit_length(samples)
        extra_loss = self.loss_function(self.ctc_logits, samples, logit_length)
        self.metric(self.ctc_logits, samples, logit_length)

        main_loss, metrics = self.model.get_loss(logits, samples, training=training)
        mtl_weight = self.hparams.mtl_weight
        loss = mtl_weight * main_loss + (1.0 - mtl_weight) * extra_loss
        metrics[self.metric.name] = self.metric.result()
        return loss, metrics

    def compute_logit_length(self, samples):
        """ compute the logit length """
        return self.model.compute_logit_length(samples)

    def reset_metrics(self):
        """ reset the metrics """
        self.metric.reset_states()
        self.model.reset_metrics()

    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        """ A more general-purpose interface for pretrained model restoration

	    :param pretrained_model: checkpoint path of mpc model
	    :param model_type: the type of pretrained model to restore
	    """
        self.model.restore_from_pretrained_model(pretrained_model, model_type)

    def decode(self, samples, hparams):
        """ beam search decoding """
        encoder_output, input_mask = self.model.decode(samples, hparams, return_encoder=True)
        # init op
        last_predictions = tf.ones([1], dtype=tf.int32) * self.sos
        history_predictions = tf.TensorArray(
            tf.int32, size=1, dynamic_size=True, clear_after_read=False
        )
        history_predictions.write(0, last_predictions)
        history_predictions = history_predictions.stack()
        init_cand_states = [history_predictions]
        step = 0
        beam_size = 1 if not hparams.beam_search else hparams.beam_size
        beam_search_decoder = BeamSearchDecoder(
            self.num_classes, self.sos, self.eos, beam_size=beam_size
        )
        beam_search_decoder.build(self.model.time_propagate)
        if hparams.beam_search and hparams.ctc_weight != 0:
            ctc_scorer = CTCPrefixScorer(
                self.eos,
                ctc_beam=hparams.beam_size*2,
                num_classes=self.num_classes,
                ctc_weight=hparams.ctc_weight,
            )
            ctc_logits = self.decoder(encoder_output, training=False)
            ctc_logits = tf.math.log(tf.nn.softmax(ctc_logits))
            init_cand_states = ctc_scorer.initial_state(init_cand_states, ctc_logits)
            beam_search_decoder.add_scorer(ctc_scorer)
        if hparams.lm_weight != 0:
            if hparams.lm_path is None:
                raise ValueError("lm path should not be none")
            lm_scorer = NGramScorer(
                hparams.lm_path,
                self.sos,
                self.eos,
                self.num_classes,
                lm_weight=hparams.lm_weight,
            )
            beam_search_decoder.add_scorer(lm_scorer)
        predictions = beam_search_decoder(
            history_predictions, init_cand_states, step, (encoder_output, input_mask)
        )
        return predictions
