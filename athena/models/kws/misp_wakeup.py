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
import tensorflow as tf
import numpy as np
from .base import BaseModel
from ...utils.hparam import register_and_parse_hparams


class MISPModel(BaseModel):
    '''
    MISP challenge KWS baseline model for e2e kws"
    '''

    default_config = {
            "num_filters":64,
            "num_classes":1,
            "dnn_hidden_size": 128 
            }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        layers = tf.keras.layers
        self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.metric =  tf.keras.metrics.BinaryAccuracy()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        input_features = layers.Input(
                shape=data_descriptions.sample_shape["input"],
                dtype=tf.float32)



        inner = input_features

        inner = layers.Permute((3, 1, 2))(inner)

        tf.ensure_shape(inner, [None, 1, None, 63])

        inner = layers.Conv2D(
                filters=256,
                kernel_size=(1, 5),
                strides=(1, 2),
                padding="same",
                use_bias=True,
                data_format="channels_last")(inner)


        inner = layers.Conv2D(
                filters=512,
                kernel_size=(1, 5),
                strides=(1, 2),
                padding="same",
                use_bias=True,
                data_format="channels_last")(inner)

        tf.ensure_shape(inner, [None, 1, None, 512])

        _, rows, cols, channels = inner.get_shape().as_list()

        inner = layers.Reshape((-1, channels))(inner)

        tf.ensure_shape(inner, [None, None, 512])

        forward_layer = layers.RNN(
                cell=layers.GRUCell(128),
                return_sequences=True)
        backward_layer = layers.RNN(
                cell=layers.GRUCell(128),
                return_sequences=True,
                go_backwards=True)
        inner = layers.Bidirectional(forward_layer, backward_layer=backward_layer,
                merge_mode='concat')(inner)

        tf.ensure_shape(inner, [None, None, 256])

        _, steps, features = inner.get_shape().as_list()


        inner = layers.Reshape((-1, features, 1))(inner)

        tf.ensure_shape(inner, [None, None, 256, 1])

        inner = layers.Conv2D(
                filters=32,
                kernel_size=(5, 5),
                strides=(1, 1),
                padding="same",
                use_bias=True,
                data_format="channels_last")(inner)

        inner = layers.Conv2D(
                filters=128,
                kernel_size=(5, 5),
                strides=(2, 2),
                padding="same",
                use_bias=True,
                data_format="channels_last")(inner)

        tf.ensure_shape(inner, [None, None, 128, 128])

        inner = layers.MaxPool2D(
                pool_size=(2, 2),
                strides=(1, 2),
                padding='same')(inner)

        inner = layers.Conv2D(
                filters=256,
                kernel_size=(5, 5),
                strides=(1, 2),
                padding="same",
                use_bias=True,
                data_format="channels_last")(inner)

        tf.ensure_shape(inner, [None, None, 32, 256])

        inner = layers.GlobalAveragePooling2D()(inner)

        tf.ensure_shape(inner, [None, 256])

        inner = layers.Dense(512)(inner)
        inner = layers.Dropout(0.2)(inner)
        inner = layers.Dense(256)(inner)
        inner = layers.Dense(1)(inner)

        self.net = tf.keras.Model(inputs=input_features, outputs=inner, name="kws_model")
        self.net.summary()
        self.tflite_model = self.build_model(data_descriptions)

    def call(self, samples, training=None):
        return self.net(samples["input"], training=training)
    
    def build_model(self, data_descriptions):
        input_features = tf.keras.layers.Input(
                shape= data_descriptions.sample_shape["input"],
                dtype=tf.float32,
                name="input_features")
        logits = self.net(input_features)
        prob = tf.math.sigmoid(logits, name="sigmoid")
        tflite_model = tf.keras.Model(inputs=input_features, outputs=prob, name="kws")
        return tflite_model


