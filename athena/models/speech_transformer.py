# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Xiaoning Lei
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
# Only support eager mode
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes

""" speech transformer implementation"""

from absl import logging
import tensorflow as tf
from .base import BaseModel
from ..loss import Seq2SeqSparseCategoricalCrossentropy
from ..metrics import Seq2SeqSparseCategoricalAccuracy
from ..utils.misc import generate_square_subsequent_mask, insert_sos_in_labels
from ..layers.commons import PositionalEncoding
from ..layers.transformer import Transformer
from ..utils.hparam import register_and_parse_hparams
from ..tools.beam_search import BeamSearchDecoder
from ..tools.lm_scorer import NGramScorer, RNNScorer


class SpeechTransformer(BaseModel):
    """ Standard implementation of a SpeechTransformer. Model mainly consists of three parts:
    the x_net for input preparation, the y_net for output preparation and the transformer itself
    """
    default_config = {
        "return_encoder_output": False,
        "num_filters": 512,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 12,
        "num_decoder_layers": 6,
        "dff": 1280,
        "rate": 0.1,
        "schedual_sampling_rate": 0.9,
        "label_smoothing_rate": 0.0
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

        # for the x_net
        num_filters = self.hparams.num_filters
        d_model = self.hparams.d_model
        layers = tf.keras.layers
        input_features = layers.Input(shape=data_descriptions.sample_shape["input"], dtype=tf.float32)
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

        # y_net for target
        input_labels = layers.Input(shape=data_descriptions.sample_shape["output"], dtype=tf.int32)
        inner = layers.Embedding(self.num_class, d_model)(input_labels)
        inner = PositionalEncoding(d_model, scale=True)(inner)
        inner = layers.Dropout(self.hparams.rate)(inner)
        self.y_net = tf.keras.Model(inputs=input_labels, outputs=inner, name="y_net")
        print(self.y_net.summary())

        # transformer layer
        self.transformer = Transformer(
            self.hparams.d_model,
            self.hparams.num_heads,
            self.hparams.num_encoder_layers,
            self.hparams.num_decoder_layers,
            self.hparams.dff,
            self.hparams.rate,
        )

        # last layer for output
        self.final_layer = layers.Dense(self.num_class, input_shape=(d_model,))

        # some temp function
        self.random_num = tf.random_uniform_initializer(0, 1)

    def call(self, samples, training: bool = None):
        x0 = samples["input"]
        y0 = insert_sos_in_labels(samples["output"], self.sos)
        x = self.x_net(x0, training=training)
        y = self.y_net(y0, training=training)
        input_length = self.compute_logit_length(samples)
        input_mask, output_mask = self._create_masks(x, input_length, y0)
        y, encoder_output = self.transformer(
            x,
            y,
            input_mask,
            output_mask,
            input_mask,
            training=training,
            return_encoder_output=True,
        )
        y = self.final_layer(y)
        if self.hparams.return_encoder_output:
            return y, encoder_output
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

    def time_propagate(self, history_logits, history_predictions, step, enc_outputs):
        """ TODO: doctring
        last_predictions: the predictions of last time_step, [beam_size]
        history_predictions: the predictions of history from 0 to time_step,
            [beam_size, time_steps]
        states: (step)
        """
        # merge
        (encoder_output, memory_mask) = enc_outputs
        step = step + 1
        output_mask = generate_square_subsequent_mask(step)
        # propagate 1 step
        logits = self.y_net(tf.transpose(history_predictions.stack()), training=False)
        logits = self.transformer.decoder(
            logits,
            encoder_output,
            tgt_mask=output_mask,
            memory_mask=memory_mask,
            training=False,
        )
        logits = self.final_layer(logits)
        logits = logits[:, -1, :]
        history_logits = history_logits.write(step - 1, logits)
        return logits, history_logits, step

    def decode(self, samples, hparams, lm_model=None, return_encoder=False):
        """ beam search decoding """
        x0 = samples["input"]
        batch = tf.shape(x0)[0]
        x = self.x_net(x0, training=False)
        input_length = self.compute_logit_length(samples)
        input_mask, _ = self._create_masks(x, input_length, None)
        encoder_output = self.transformer.encoder(x, input_mask, training=False)
        if return_encoder:
            return encoder_output, input_mask
        # init op
        last_predictions = tf.ones([batch], dtype=tf.int32) * self.sos
        history_predictions = tf.TensorArray(
            tf.int32, size=1, dynamic_size=True, clear_after_read=False
        )
        step = 0
        history_predictions.write(0, last_predictions)
        history_predictions = history_predictions.stack()
        init_cand_states = [history_predictions]

        beam_size = 1 if not hparams.beam_search else hparams.beam_size
        beam_search_decoder = BeamSearchDecoder(
            self.num_class, self.sos, self.eos, beam_size=beam_size
        )
        beam_search_decoder.build(self.time_propagate)
        if hparams.lm_weight != 0:
            if hparams.lm_path is None:
                raise ValueError("lm path should not be none")
            if hparams.lm_type == "ngram":
                lm_scorer = NGramScorer(
                    hparams.lm_path,
                    self.sos,
                    self.eos,
                    self.num_class,
                    lm_weight=hparams.lm_weight,
                )
            elif hparams.lm_type == "rnn":
                lm_scorer = RNNScorer(
                    lm_model,
                    lm_weight=hparams.lm_weight)
            beam_search_decoder.add_scorer(lm_scorer)
        predictions = beam_search_decoder(
            history_predictions, init_cand_states, step, (encoder_output, input_mask)
        )
        return predictions

    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        if model_type == "":
            return
        if model_type == "mpc":
            logging.info("loading from pretrained mpc model")
            self.x_net = pretrained_model.x_net
            self.transformer.encoder = pretrained_model.encoder
        elif model_type == "SpeechTransformer":
            logging.info("loading from pretrained SpeechTransformer model")
            self.x_net = pretrained_model.x_net
            self.y_net = pretrained_model.y_net
            self.transformer = pretrained_model.transformer
            self.final_layer = pretrained_model.final_layer
        else:
            raise ValueError("NOT SUPPORTED")


class SpeechTransformer2(SpeechTransformer):
    """ Decoder for SpeechTransformer2 works for two pass schedual sampling"""

    def call(self, samples, training: bool = None):
        x0 = samples["input"]
        y0 = insert_sos_in_labels(samples["output"], self.sos)
        x = self.x_net(x0, training=training)
        y = self.y_net(y0, training=training)
        input_length = self.compute_logit_length(samples)
        input_mask, output_mask = self._create_masks(x, input_length, y0)
        # first pass
        y, encoder_output = self.transformer(
            x,
            y,
            input_mask,
            output_mask,
            input_mask,
            training=training,
            return_encoder_output=True,
        )
        y_pre = self.final_layer(y)
        # second pass
        y = self.mix_target_sequence(y0, y_pre, training)
        y, encoder_output = self.transformer(
            x,
            y,
            input_mask,
            output_mask,
            input_mask,
            training=training,
            return_encoder_output=True,
        )
        y = self.final_layer(y)
        if self.hparams.return_encoder_output:
            return y, encoder_output
        return y

    def mix_target_sequence(self, gold_token, predicted_token, training, top_k=5):
        """ to mix gold token and prediction
        param gold_token: true labels
        param predicted_token: predictions by first pass
        return: mix of the gold_token and predicted_token
        """
        mix_result = tf.TensorArray(
            tf.float32, size=1, dynamic_size=True, clear_after_read=False
        )
        for i in tf.range(tf.shape(gold_token)[-1]):
            if self.random_num([1]) > self.hparams.schedual_sampling_rate:# do schedual sampling
                selected_input = predicted_token[:, i, :]
                selected_idx = tf.nn.top_k(selected_input, top_k).indices
                embedding_input = self.y_net.layers[1](selected_idx, training=training)
                embedding_input = tf.reduce_mean(embedding_input, axis=1)
                mix_result = mix_result.write(i, embedding_input)
            else:
                selected_input = tf.reshape(gold_token[:, i], [-1, 1])
                embedding_input = self.y_net.layers[1](selected_input, training=training)
                mix_result = mix_result.write(i, embedding_input[:, 0, :])
        final_input = self.y_net.layers[2](tf.transpose(mix_result.stack(), [1, 0, 2]),
                                           training=training)
        final_input = self.y_net.layers[3](final_input, training=training)
        return final_input
