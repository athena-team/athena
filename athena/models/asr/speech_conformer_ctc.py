# coding=utf-8
# Copyright (C) ATHENA AUTHORS;
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
from typing import List

from absl import logging
import tensorflow as tf
from athena.models.base import BaseModel
from athena.loss import CTCLoss
from athena.metrics import CTCAccuracy
from athena.utils.misc import generate_square_subsequent_mask, insert_sos_in_labels, create_multihead_mask, mask_finished_preds, mask_finished_scores
from athena.layers.commons import PositionalEncoding
from athena.layers.transformer import Transformer
from athena.layers.conformer_ctc import ConformerCTC
from athena.utils.hparam import register_and_parse_hparams
from athena.tools.ctc_scorer import CTCPrefixScoreTH
from athena.tools.ctc_decoder import ctc_prefix_beam_decoder

class SpeechConformerCTC(BaseModel):
    """ Standard implementation of a SpeechTransformer. Model mainly consists of three parts:
    the x_net for input preparation and the transformer itself
    """
    default_config = {
        "return_encoder_output": False,
        "num_filters": 512,
        "d_model": 512,
        "num_heads": 8,
        "cnn_module_kernel": 15,
        "num_encoder_layers": 12,
        "dff": 1280,
        "max_position": 1600,
        "dropout_rate": 0.1,
        "schedual_sampling_rate": 0.9,
        "label_smoothing_rate": 0.0,
        "unidirectional": False,
        "look_ahead": 0,
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        self.num_class = data_descriptions.num_class + 1
        self.sos = self.num_class - 1
        self.eos = self.num_class - 1
        ls_rate = self.hparams.label_smoothing_rate
        self.loss_function = CTCLoss(blank_index=-1)
        self.metric = CTCAccuracy()

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
        inner = PositionalEncoding(d_model, max_position=self.hparams.max_position, scale=False)(inner)
        inner = layers.Dropout(self.hparams.dropout_rate)(inner)
        self.x_net = tf.keras.Model(inputs=input_features, outputs=inner, name="x_net")
        print(self.x_net.summary())

        # Conformer encoder layers
        self.conformer_ctc = ConformerCTC(
            self.hparams.d_model,
            self.hparams.num_heads,
            self.hparams.cnn_module_kernel,
            self.hparams.num_encoder_layers,
            self.hparams.dff,
            self.hparams.dropout_rate,
        )

        # last layer for output
        self.layer_norm = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.layer_dense = layers.Dense(self.num_class, input_shape=(d_model,))
        self.softmax = layers.Softmax(name='softmax')
        # some temp function
        self.random_num = tf.random_uniform_initializer(0, 1)

    def call(self, samples, training: bool = None):
        x0 = samples["input"]
        y0 = insert_sos_in_labels(samples["output"], self.sos)
        x = self.x_net(x0, training=training)
        input_length = self.compute_logit_length(samples["input_length"])
        input_mask, output_mask = create_multihead_mask(x, input_length, y0)
        y = self.conformer_ctc(
            x,
            input_mask,
            training=training,
            return_encoder_output=True,
        )
        y = self.layer_norm(y)
        y = self.layer_dense(y)
        return y
 
    def compute_logit_length(self, input_length):
        """ used for get logit length """
        input_length = tf.cast(input_length, tf.float32)
        logit_length = tf.math.ceil(input_length / 2)
        logit_length = tf.math.ceil(logit_length / 2)
        logit_length = tf.cast(logit_length, tf.int32)
        return logit_length

    def _forward_encoder(self, speech, speech_length, training=None):
        x = self.x_net(speech, training=training)
        input_length = self.compute_logit_length(speech_length)
        input_mask, _ = create_multihead_mask(x, input_length, None)
        # 1. Encoder
        encoder_output = self.conformer_ctc.encoder(x, input_mask, training=training)  # (B, maxlen, encoder_dim)
        return encoder_output, input_mask
    
    def _forward_encoder_log_ctc(self, samples, training: bool = None):
        speech = samples['input']
        speech_length = samples['input_length']

        x = self.x_net(speech, training=training)
        input_length = self.compute_logit_length(speech_length)
        input_mask, _ = create_multihead_mask(x, input_length, None)
        encoder_output = self.conformer_ctc.encoder(x, input_mask, training=training)  # (B, maxlen, encoder_dim)
        predictions = self.layer_norm(encoder_output)
        predictions = self.layer_dense(predictions)
        predictions = tf.nn.log_softmax(predictions)
        return predictions, input_mask

    def decode(self, samples, hparams, lm_model=None):
        """
        Initialization of the model for decoding,
        decoder is called here to create predictions

        Args:
            samples: the data source to be decoded
            hparams: decoding configs are included here
            lm_model: lm model
        Returns::

            predictions: the corresponding decoding results
        """
        if hparams.decoder_type == "argmax":
            predictions = self.argmax(samples, hparams)
        elif hparams.decoder_type == "ctc_prefix_beam_search":
            predictions = self.ctc_prefix_beam_search(samples, hparams, self.layer_dense)
        else:
            logging.warning('Unsupport decoder type: {}'.format(hparams.decoder_type))
        return predictions
    
    def argmax(self, samples, hparams):
        """
        argmax for the Conformer CTC model

        Args:
            samples: the data source to be decoded
            hparams: decoding configs are included here
        Returns::
            predictions: the corresponding decoding results
        """
        x0 = self.x_net(samples['input'], training=False)
        encoder_output = self.conformer_ctc.encoder(x0, training=False)
        predictions = self.layer_norm(encoder_output)
        predictions = self.layer_dense(predictions)
        predictions = self.softmax(predictions)
        top1 = tf.math.argmax(predictions, axis=-1)
        top1 = self.merge_ctc_sequence(top1, blank=self.sos)
        return top1
    
    def ctc_prefix_beam_search(
            self, samples, hparams, ctc_final_layer
    ) -> List[int]:
        speech = samples["input"]
        speech_lengths = samples["input_length"]
        beam_size, ctc_weight = hparams.beam_size, hparams.ctc_weight
        assert speech.shape[0] == speech_lengths.shape[0]

        encoder_out, encoder_mask = self._forward_encoder(speech, speech_lengths, training=False)
        encoder_out = self.layer_norm(encoder_out)

        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out = ctc_prefix_beam_decoder(
            encoder_out, ctc_final_layer, beam_size, blank_id=self.sos)

        return tf.convert_to_tensor(hyps[0][0])[tf.newaxis, :]

    def freeze_ctc_prefix_beam_search(
            self, samples, ctc_final_layer, hparams=None,beam_size=1
    ) -> List[int]:
        speech = samples["input"]
        speech_lengths = samples["input_length"]
        x = self.x_net(speech, training=False)
        input_length = self.compute_logit_length(speech_lengths)
        input_mask, _ = create_multihead_mask(x, input_length, None)
        # 1. Encoder
        encoder_output = self.transformer.encoder(x, input_mask, training=False)

        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        # hyps, encoder_out = ctc_prefix_beam_decoder(
        #     encoder_out, ctc_final_layer, beam_size, blank_id=self.sos)  # tf.nn.ctc_beam_search_decoder
        # return tf.convert_to_tensor(hyps[0][0])[tf.newaxis, :]

        ctc_probs = tf.nn.log_softmax(ctc_final_layer(encoder_output), axis=2)  # (1, maxlen, vocab_size)
        ctc_probs = tf.transpose(ctc_probs, (1, 0, 2))
        decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(  # (max_time, batch_size, num_classes)
            ctc_probs, input_length, beam_width=beam_size, top_paths=beam_size
        )
        return decoded[0].values[tf.newaxis, :]
    
    def merge_ctc_sequence(self, seqs, blank=-1):
        index = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        # Blank is always at the beginning
        index = index.write(0, tf.not_equal(blank, seqs[:,0]) & tf.not_equal(0, seqs[:,0]))
        for i in tf.range(1, tf.shape(seqs)[1]):
            is_keep = tf.not_equal(blank, seqs[:,i]) & tf.not_equal(0, seqs[:,i]) & tf.not_equal(seqs[:,i-1], seqs[:,i])
            index = index.write(i, is_keep)
        index = tf.transpose(index.stack(), [1, 0]) # [length, batch] -> [batch, length]

        validate_seqs = tf.where(index, seqs, tf.zeros_like(seqs))
        validate_seqs = tf.sparse.from_dense(validate_seqs)
        return validate_seqs

    def freeze_beam_search(self, samples, beam_size):
        """ beam search for freeze only support batch=1

        Args:
            samples: the data source to be decoded
            beam_size: beam size
        """
        x0 = self.x_net(samples['input'], training=False)
        encoder_output = self.conformer_ctc.encoder(x0, training=False)
        predictions = self.layer_norm(encoder_output)
        predictions = self.layer_dense(predictions)
        predictions = self.softmax(predictions)
        y0 = tf.math.argmax(predictions, output_type=tf.dtypes.int32, axis=-1)
        #y0 = self.merge_ctc_sequence(y0, blank=self.sos)
        return {'output_0': y0}

    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        if model_type == "":
            return
        if model_type == "mpc":
            logging.info("loading from pretrained mpc model")
            self.x_net = pretrained_model.x_net
            self.transformer.encoder = pretrained_model.encoder
        elif model_type == "SpeechConformer":
            logging.info("loading from pretrained SpeechConformer model")
            self.x_net = pretrained_model.x_net
            self.y_net = pretrained_model.y_net
            self.transformer = pretrained_model.transformer
            self.final_layer = pretrained_model.final_layer
        else:
            raise ValueError("NOT SUPPORTED")


