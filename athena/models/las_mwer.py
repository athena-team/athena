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

""" LAS model with MWER loss implementation"""
from absl import logging
import tensorflow as tf
from .base import BaseModel
from ..loss import Seq2SeqSparseCategoricalCrossentropy, CTCLoss
from ..metrics import Seq2SeqSparseCategoricalAccuracy, CTCAccuracy
from ..utils.misc import generate_square_subsequent_mask, insert_sos_in_labels
from ..layers.commons import PositionalEncoding

from ..utils.misc import validate_seqs
from ..layers.listen_attend_spell import LASLayer
from ..layers.batch_beam_search import BatchBeamSearchLayer
from ..utils.hparam import register_and_parse_hparams


class SpeechLASMwer(BaseModel):
    """ Standard implementation of a LAS model with MWER. Model mainly consists of three parts:
    the x_net for input preparation, the y_net for output preparation and the transformer itself
    """
    default_config = {
        "num_filters": 512,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 12,
        "decoder_dim": 320,
        "attention_dim": 128,
        "dff": 1280,
        "rate": 0.1,
        "schedual_sampling_rate": 0.8,
        "label_smoothing_rate": 0.0,
        "mwer_weight": 0.01,
        "beam_size": 4
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        self.num_class = data_descriptions.num_class + 1
        self.sos = self.num_class - 1
        self.eos = self.num_class - 1
        ls_rate = self.hparams.label_smoothing_rate
        self.loss_function = Seq2SeqSparseCategoricalCrossentropy(
            num_classes=self.num_class, eos=self.eos, label_smoothing=ls_rate
        )
        self.metric = Seq2SeqSparseCategoricalAccuracy(eos=self.eos, name="Accuracy")
        self.ctc_loss = CTCLoss(blank_index=-1)
        self.ctc_metric = CTCAccuracy()
        self.inputs_mask = None
        self.batch_size = None
        self.encoder_output = None
        self.beam_size = self.hparams.beam_size

        self.beam_search_layer = BatchBeamSearchLayer(
            self.num_class, self.sos, self.eos, self.beam_size,
            self.hparams.attention_dim, self.hparams.decoder_dim
        )
        self.beam_search_layer.set_one_step(self.batch_time_propagate)
        # for the x_net
        num_filters = self.hparams.num_filters
        d_model = self.hparams.d_model
        layers = tf.keras.layers
        input_features = layers.Input(shape=data_descriptions.sample_shape["input"],
                                      dtype=tf.float32)
        inner = layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(input_features)
        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu6(inner)
        inner = layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(inner)
        inner = layers.BatchNormalization()(inner)

        inner = tf.nn.relu6(inner)
        _, _, dim, channels = inner.get_shape().as_list()
        output_dim = dim * channels
        inner = layers.Reshape((-1, output_dim))(inner)

        inner = layers.Dense(d_model, activation=tf.nn.relu6)(inner)
        inner = PositionalEncoding(d_model, scale=False)(inner)
        inner = layers.Dropout(self.hparams.rate)(inner)  # self.hparams.rate
        self.x_net = tf.keras.Model(inputs=input_features, outputs=inner, name="x_net")
        print(self.x_net.summary())

        self.las = LASLayer(
            self.hparams.d_model,
            self.hparams.num_heads,
            self.hparams.num_encoder_layers,
            self.hparams.dff,
            self.hparams.decoder_dim,
            self.hparams.attention_dim,
            self.num_class,
            self.hparams.rate
        )

        # some temp function
        self.random_num = tf.random_uniform_initializer(0, 1)
        self.ctc_decoder = layers.Dense(self.num_class)
        self.ctc_logits = None

    def call(self, samples, training: bool = None):
        x0 = samples["input"]
        y0 = insert_sos_in_labels(samples["output"], self.sos)
        x = self.x_net(x0, training=training)
        y = tf.one_hot(y0, self.num_class)
        input_length = self.compute_logit_length(samples)
        self.inputs_mask, output_mask = self._create_masks(x, input_length, y0)

        y, self.encoder_output = self.las(
            x,
            y,
            self.inputs_mask,
            return_encoder_output=True,
            schedual_sampling_rate=self.hparams.schedual_sampling_rate,
            training=training
        )
        self.ctc_logits = self.ctc_decoder(self.encoder_output, training=training)
        return y

    @staticmethod
    def _create_masks(x, input_length, y):
        r""" Generate a square mask for the sequence. The masked positions are
        filled with float(1.0). Unmasked positions are filled with float(0.0).
        """
        input_mask, output_mask = None, None
        if x is not None:
            input_mask = 1.0 - tf.sequence_mask(
                input_length, tf.shape(x)[1], dtype=tf.float32
            )
            input_mask = input_mask[:, tf.newaxis, tf.newaxis, :]
            input_mask.set_shape([None, None, None, None])
        if y is not None:
            output_mask = tf.cast(tf.math.equal(y, 0), tf.float32)
            output_mask = output_mask[:, tf.newaxis, tf.newaxis, :]
            look_ahead_mask = generate_square_subsequent_mask(tf.shape(y)[1])
            output_mask = tf.maximum(output_mask, look_ahead_mask)
            output_mask.set_shape([None, None, None, None])
        return input_mask, output_mask

    def compute_logit_length(self, samples):
        """ used for get logit length """
        input_length = tf.cast(samples["input_length"], tf.float32)
        logit_length = tf.math.ceil(input_length / 2)
        logit_length = tf.math.ceil(logit_length / 2)
        logit_length = tf.cast(logit_length, tf.int32)
        return logit_length

    def batch_time_propagate(self, history_logits_array, history_predictions_array,
                             step, decoder_states, enc_outputs, training=False):
        """get one step result in training"""
        (encoder_output, _) = enc_outputs

        history_predictions = tf.transpose(history_predictions_array.stack(), [1, 2, 0])[:, :, -1:]
        if step == 0:
            history_predictions = history_predictions[:, :1, :]
        history_predictions = tf.reshape(history_predictions, [-1, 1])

        y = tf.one_hot(history_predictions, self.num_class)
        logits, decoder_states = self.las.decoder(y,
                                                  memory=encoder_output,
                                                  training=training,
                                                  states=decoder_states,
                                                  return_states=True)
        logits = logits[:, -1, :]
        logits = tf.reshape(logits, [self.batch_size, -1, self.num_class])
        history_logits_array = history_logits_array.write(step, logits)
        return logits, history_logits_array, decoder_states

    def time_propagate(self, history_logits, history_predictions,
                       step, decoder_states, enc_outputs):
        """ TODO: doctring
        last_predictions: the predictions of last time_step, [beam_size]
        history_predictions: the predictions of history from 0 to time_step,
            [beam_size, time_steps]
        states: (step)
        """
        # merge
        (encoder_output, _) = enc_outputs
        step = step + 1
        # propagate 1 step
        y = tf.one_hot(tf.transpose(history_predictions.stack())[:, -1:], self.num_class)
        logits, decoder_states = self.las.decoder(y,
                                                  memory=encoder_output,
                                                  training=False,
                                                  states=decoder_states,
                                                  return_states=True)

        logits = logits[:, -1, :]  # [batch, n_num]
        history_logits = history_logits.write(step - 1, logits)
        return logits, history_logits, step, decoder_states

    def get_loss(self, outputs, samples, training=None):
        logit_length = self.compute_logit_length(samples)
        main_loss = self.loss_function(outputs, samples, logit_length)
        ctc_loss = self.ctc_loss(self.ctc_logits, samples, logit_length)
        extra_loss = self.get_decode_loss(samples)
        loss = self.hparams.mwer_weight * main_loss + \
               extra_loss + self.hparams.mwer_weight * ctc_loss
        self.metric(outputs, samples, logit_length)
        self.ctc_metric(self.ctc_logits, samples, logit_length)
        metrics = {self.metric.name: self.metric.result(),
                   self.ctc_metric.name: self.ctc_metric.result()}
        return loss, metrics

    def get_word_error(self, samples, predictions):
        """get word error between sample and prediction"""
        output_beam = self.hparams.beam_size
        sparse_predictions = validate_seqs(predictions, self.eos)[0]
        output = samples["output"]
        batch_size = tf.shape(output)[0]
        output_tile = tf.tile(output, [1, output_beam])
        output = tf.reshape(output_tile, [batch_size*output_beam, -1])
        validated_label = tf.cast(
            tf.sparse.from_dense(output), dtype=tf.int64
        )
        word_error = tf.edit_distance(
            sparse_predictions, validated_label, normalize=False
        )
        return word_error

    def get_decode_loss(self, samples):
        """get MWER loss"""
        self.batch_size = tf.shape(samples["input"])[0]

        predictions, output_scores = self.beam_search_layer(
            (self.encoder_output, self.inputs_mask),
            output_beam=self.beam_size, training=True
        )
        word_error = self.get_word_error(samples, predictions)
        output_prob = tf.exp(output_scores)
        output_prob_div = tf.reduce_sum(output_prob, -1, keepdims=True)
        output_prob_renorm = output_prob / output_prob_div
        word_error = tf.reshape(word_error, [self.batch_size, -1])
        word_error_sub = tf.reduce_mean(word_error, -1, keepdims=True)
        word_error -= word_error_sub
        loss = tf.reduce_sum(tf.abs(output_prob_renorm * word_error), -1)
        loss = tf.reduce_mean(loss)
        return loss

    def reset_metrics(self):
        self.metric.reset_states()
        self.ctc_metric.reset_states()

    def decode(self, samples, hparams, decoder, return_encoder=False):
        """ beam search decoding """
        x0 = samples["input"]
        batch_size = tf.shape(x0)[0]
        x = self.x_net(x0, training=False)
        input_length = self.compute_logit_length(samples)
        input_mask, _ = self._create_masks(x, input_length, None)
        encoder_output = self.las.encoder(x, src_mask=input_mask, training=False)
        if return_encoder:
            return encoder_output, input_mask
        history_predictions = tf.ones([batch_size, 1], dtype=tf.int32) * self.sos
        init_cand_states = [history_predictions]
        step = 0

        predictions = decoder(
            history_predictions, init_cand_states, step, (encoder_output, input_mask)
        )
        return predictions

    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        logging.info("loading from pretrained mtl model")
        self.x_net = pretrained_model.x_net
        self.las = pretrained_model.las
        self.ctc_decoder = pretrained_model.ctc_decoder
