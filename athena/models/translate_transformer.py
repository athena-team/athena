# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li
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
# pylint: disable=no-member, invalid-name
r""" an example of seq2seq model; NeuralTranslateTransformer using transformer
"""

import tensorflow as tf
from .base import BaseModel
from ..loss import Seq2SeqSparseCategoricalCrossentropy
from ..metrics import Seq2SeqSparseCategoricalAccuracy
from ..layers.transformer import Transformer
from ..layers.commons import PositionalEncoding
from ..utils.misc import generate_square_subsequent_mask, insert_sos_in_labels
from ..utils.hparam import register_and_parse_hparams


class NeuralTranslateTransformer(BaseModel):
    ''' This is an example of seq2seq model using transformer '''
    default_config = {
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 12,
        "num_decoder_layers": 6,
        "dff": 2048,
        "rate": 0.1,
        "label_smoothing_rate": 0.0
    }
    def __init__(self, data_descriptions, config=None):
        super().__init__()
        p = register_and_parse_hparams(self.default_config, config)

        self.sos = data_descriptions.num_class
        self.eos = data_descriptions.num_class

        self.loss_function = Seq2SeqSparseCategoricalCrossentropy(
            num_classes=data_descriptions.num_class + 1,
            eos=self.eos,
            label_smoothing=p.label_smoothing_rate
        )
        self.metric = Seq2SeqSparseCategoricalAccuracy(eos=self.eos, name="Accuracy")

        layers = tf.keras.layers
        input_labels = layers.Input(shape=data_descriptions.sample_shape["input"], dtype=tf.int32)
        inner = layers.Embedding(data_descriptions.input_vocab_size, p.d_model)(input_labels)
        inner = PositionalEncoding(p.d_model, scale=True)(inner)
        inner = layers.Dropout(p.rate)(inner)
        self.x_net = tf.keras.Model(inputs=input_labels, outputs=inner, name="x_net")

        output_labels = layers.Input(shape=data_descriptions.sample_shape["output"], dtype=tf.int32)
        inner = layers.Embedding(data_descriptions.num_class + 1, p.d_model)(output_labels)
        inner = PositionalEncoding(p.d_model, scale=True)(inner)
        inner = layers.Dropout(p.rate)(inner)
        self.y_net = tf.keras.Model(inputs=output_labels, outputs=inner, name="y_net")

        # transformer network
        self.transformer = Transformer(
            p.d_model,
            p.num_heads,
            p.num_encoder_layers,
            p.num_decoder_layers,
            p.dff,
            p.rate)
        self.final_layer = tf.keras.layers.Dense(data_descriptions.num_class + 1)

    def call(self, samples, training=None):
        x = samples["input"]
        y = insert_sos_in_labels(samples["output"], self.sos)
        input_mask, output_mask = self._create_masks(x, y)

        x = self.x_net(x, training=training)
        y = self.y_net(y, training=training)

        y = self.transformer(
            x,
            y,
            input_mask,
            output_mask,
            input_mask,
            training=training
        )
        y = self.final_layer(y)
        return y

    @staticmethod
    def _create_masks(x, y):
        def create_padding_mask(seq):
            seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
            return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
        input_mask = create_padding_mask(x)
        look_ahead_mask = generate_square_subsequent_mask(tf.shape(y)[1])
        dec_target_padding_mask = create_padding_mask(y)
        output_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return input_mask, output_mask
