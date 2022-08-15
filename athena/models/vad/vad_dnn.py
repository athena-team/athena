# coding=utf-8
# Copyright (C) 2022 ATHENA AUTHORS; Tingwei Guo; Shuaijiang Zhao
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
import sys
import tensorflow as tf
import numpy as np
from athena.models.base import BaseModel
from athena.utils.hparam import register_and_parse_hparams
from absl import logging

class VadDnn(BaseModel):
    '''
    implementation of a frame level or segment speech classification
    '''

    default_config = {
            "num_classes":2,
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
        _, _, dim, channels = input_features.get_shape().as_list()
        output_dim = dim * channels
        inner = layers.Reshape((-1, output_dim))(input_features)
        inner = layers.Dense(
                self.hparams.dnn_hidden_size,
                activation=tf.nn.relu)(inner)
        inner = layers.Dense(
                self.hparams.dnn_hidden_size,
                activation=tf.nn.relu)(inner)
        inner = layers.Dense(
                self.hparams.dnn_hidden_size,
                activation=tf.nn.relu)(inner)
        inner = layers.Dense(
                self.hparams.dnn_hidden_size,
                activation=tf.nn.relu)(inner)

        inner = layers.Dense(self.hparams.num_classes)(inner)

        self.net = tf.keras.Model(inputs=input_features, outputs=inner, name="vad_model")
        self.net.summary()

    def call(self, samples, training=None):
        x = samples["input"]
        x = tf.reshape(x, (tf.shape(x)[0], -1))
        y = self.net(x, training=training)
        return y

    def get_loss(self, outputs, samples, training=None):
        loss = self.loss_function(samples["output"], outputs)
        self.metric.update_state(samples["output"], outputs)
        metrics = {self.metric.name: self.metric.result()}
        return loss, metrics
