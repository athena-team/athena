# coding=utf-8
# Copyright (C) 2022 ATHENA AUTHORS; Yanguang Xu; Yang Han; Jianwei Sun
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
from ...utils.hparam import register_and_parse_hparams
from ...layers.transformer import TransformerEncoderLayer, TransformerEncoder
from ...layers.commons import PositionalEncoding

class KWSTransformer(BaseModel):
    """ Standard implementation of a KWSTransformer. Model mainly consists of three parts:
    the x_net for input preparation, the transformer itself
    """
    default_config = {
        "num_classes":1,
        "num_filters": 512,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 12,
        "rate": 0.1
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        layers = tf.keras.layers
        self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
        self.metric =  tf.keras.metrics.BinaryAccuracy()

        # learnable class embedding to learn a global feature that represents the whole spectrogram
        self.class_emb = self.add_weight("class_emb", shape=(1, 1, self.hparams.d_model), initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02))

        input_features = layers.Input(
                shape=data_descriptions.sample_shape["input"],
                dtype=tf.float32)

        # tf.ensure_shape(input_features, [None, None, 63, 1])

        inner = layers.Conv2D(
            filters=self.hparams.num_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(input_features)


        # tf.ensure_shape(inner, [None, None, 32, 512])

        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu6(inner)
        inner = layers.Conv2D(
            filters=self.hparams.num_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(inner)
        inner = layers.BatchNormalization()(inner)

        # tf.ensure_shape(inner, [None, None, 16, 512])

        inner = tf.nn.relu6(inner)
        _, _, dim, channels = inner.get_shape().as_list()
        output_dim = dim * channels
        inner = layers.Reshape((-1, output_dim))(inner)
        inner = layers.Dense(self.hparams.d_model, activation=tf.nn.relu6)(inner)

        self.x_net = tf.keras.Model(inputs=input_features, outputs=inner, name="x_net")
        logging.info(self.x_net.summary())

        self.concat_layer = layers.Concatenate(axis=1)              
        self.dropout_layer = layers.Dropout(self.hparams.rate)

        # transformer encoder
        encoder_layers = [
            TransformerEncoderLayer(self.hparams.d_model, self.hparams.num_heads) for _ in range(self.hparams.num_encoder_layers)
            ]
        self.encoder = TransformerEncoder(encoder_layers)
        
        # last layer for output
        self.final_layer = layers.Dense(self.hparams.num_classes, input_shape=(self.hparams.d_model,))

        self.tflite_model = self.build_model(data_descriptions)


    def call(self, samples, training=None):
        x0 = samples["input"]
        x = self.x_net(x0, training=training)
        
        batch_size = tf.shape(x)[0]
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.hparams.d_model])
        x = self.concat_layer([class_emb, x])
        x = PositionalEncoding(self.hparams.d_model, scale=False)(x)
        x = self.dropout_layer(x,training=training)  # self.hparams.rate

        x = self.encoder(x, training=training)
        attention_output = x[:, 1:]
        class_output = x[:, 0]
        y = self.final_layer(class_output)
        return y

    def build_model(self, data_descriptions):
        input_features = tf.keras.layers.Input(
                shape= data_descriptions.sample_shape["input"],
                dtype=tf.float32,
                name="input_features")

        x = self.x_net(input_features, training=False)
        
        batch_size = tf.shape(x)[0]
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, 1, self.hparams.d_model])
        x = self.concat_layer([class_emb, x])
        x = PositionalEncoding(self.hparams.d_model, scale=False)(x)
        x = self.dropout_layer(x,training=False)  # self.hparams.rate

        x = self.encoder(x, training=False)

        class_output = x[:, 0]
        logits = self.final_layer(class_output)

        prob = tf.math.sigmoid(logits, name="sigmoid")
        tflite_model = tf.keras.Model(inputs=input_features, outputs=prob, name="kws")
        return tflite_model

