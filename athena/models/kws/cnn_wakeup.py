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
from absl import logging

class CnnModel(BaseModel):
    '''
    CNN model for kws"
    '''

    default_config = {
            "num_filters":64,
            "num_classes":9,
            "dnn_hidden_size": 128 
            }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        layers = tf.keras.layers
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric =  tf.keras.metrics.SparseCategoricalAccuracy()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        input_features = layers.Input(
                shape=data_descriptions.sample_shape["input"],
                dtype=tf.float32
                )

        inner = layers.Conv2D(
                filters=16,
                kernel_size=(8,8),
                strides=(2, 2),
                padding="same",
                use_bias=True,
                data_format="channels_last")(input_features)

        inner = layers.MaxPool2D(
                    pool_size=(2,2),
                    strides=(2,2),
                    padding='same')(inner)

        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu(inner)

        inner = layers.Conv2D(
                filters=32,
                kernel_size=(4, 4),
                strides=(2,2),
                padding="same",
                use_bias=True,
                data_format="channels_last")(inner)
        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu(inner)

        _, h, w, c = inner.get_shape().as_list()
        output_dim = h * w * c
        inner = layers.Reshape((output_dim,))(inner)
        inner = layers.Dense(
                128)(inner)
        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu(inner)

        inner = layers.Dense(
                128)(inner)
        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu(inner)

        inner = layers.Dense(self.hparams.num_classes)(inner)

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
        prob = tf.nn.softmax(logits, name="softmax")
        tflite_model = tf.keras.Model(inputs=input_features, outputs=prob, name="kws")
        return tflite_model

