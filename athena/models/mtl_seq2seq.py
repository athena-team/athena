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
""" a implementation of multi-task model with attention and ctc loss """

import io
import tensorflow as tf
from tensorflow.keras.layers import Dense
from .base import BaseModel
from ..loss import CTCLoss
from ..metrics import CTCAccuracy
from .speech_transformer import SpeechTransformer, SpeechTransformer2
from ..utils.hparam import register_and_parse_hparams


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

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.num_class = data_descriptions.num_class + 1
        self.sos = self.num_class - 1
        self.eos = self.num_class - 1

        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        self.loss_function = CTCLoss(blank_index=-1)
        self.metric = CTCAccuracy()
        self.model = self.SUPPORTED_MODEL[self.hparams.model](
            data_descriptions, self.hparams.model_config
        )
        self.time_propagate = self.model.time_propagate
        self.decoder = Dense(self.num_class)

        # for deployment
        self.deploy_encoder = None
        self.deploy_decoder = None

        # for WFST
        self.vocab = {}
        for line in io.open(data_descriptions.hparams.text_config["model"], 'r', encoding='utf-8').readlines():
            char, idx = line.strip().split()[0], line.strip().split()[1]
            self.vocab[char] = int(idx)
        # we need dict of words for WFST decoding
        if data_descriptions.hparams.words is not None:
            self.words = []
            for line in io.open(data_descriptions.hparams.words, 'r', encoding='utf-8').readlines():
                word = line.strip().split()[0]
                self.words.append(word)

    def call(self, samples, training=None):
        """ call function in keras layers """
        attention_logits, encoder_output = self.model(samples, training=training)
        ctc_logits = self.decoder(encoder_output, training=training)
        return attention_logits, ctc_logits

    def get_loss(self, outputs, samples, training=None):
        """ get loss used for training """
        attention_logits, ctc_logits = outputs
        logit_length = self.compute_logit_length(samples)
        extra_loss = self.loss_function(ctc_logits, samples, logit_length)
        self.metric(ctc_logits, samples, logit_length)

        main_loss, metrics = self.model.get_loss(attention_logits, samples, training=training)
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

	    Args:
	        pretrained_model: checkpoint path of mpc model
	        model_type: the type of pretrained model to restore
	    """
        self.model.restore_from_pretrained_model(pretrained_model, model_type)

    def decode(self, samples, hparams, decoder):
        """
        Initialization of the model for decoding,
        decoder is called here to create predictions

        Args:
            samples: the data source to be decoded
            hparams: decoding configs are included here
            decoder: it contains the main decoding operations
        Returns::

            predictions: the corresponding decoding results
        """
        encoder_output, input_mask = self.model.decode(samples, hparams, decoder, return_encoder=True)
        # init op
        last_predictions = tf.ones([1], dtype=tf.int32) * self.sos
        history_predictions = tf.TensorArray(
            tf.int32, size=1, dynamic_size=True, clear_after_read=False
        )
        history_predictions.write(0, last_predictions)
        history_predictions = history_predictions.stack()
        init_cand_states = [history_predictions]
        step = 0
        if hparams.decoder_type == "beam_search_decoder" and hparams.ctc_weight != 0 and decoder.ctc_scorer is not None:
            ctc_logits = self.decoder(encoder_output, training=False)
            ctc_logits = tf.math.log(tf.nn.softmax(ctc_logits))
            init_cand_states = decoder.ctc_scorer.initial_state(init_cand_states, ctc_logits)
        if hparams.decoder_type == "beam_search_decoder":
            predictions = decoder(
                history_predictions, init_cand_states, step, (encoder_output, input_mask)
            )
        elif hparams.decoder_type == "wfst_decoder":
            initial_packed_states = (0,)  # this is basically decoding step
            decoder.decode((encoder_output, input_mask), initial_packed_states, self.model.inference_one_step)
            words_prediction_id = decoder.get_best_path()
            words_prediction = ''.join([self.words[int(idx)] for idx in words_prediction_id])
            predictions = [self.vocab[prediction] for prediction in words_prediction]
            predictions = tf.constant([predictions])
            predictions = tf.cast(predictions, tf.int64)
        return predictions

    def deploy(self):
        """ deployment function """
        self.model.deploy()
        self.deploy_encoder = self.model.deploy_encoder
        self.deploy_decoder = self.model.deploy_decoder
