# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Xiaoning Lei
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
""" RNN language model implementation"""

import tensorflow as tf
from .base import BaseModel
from ..utils.misc import insert_eos_in_labels, insert_sos_in_labels
from ..utils.hparam import register_and_parse_hparams
from ..layers.commons import SUPPORTED_RNNS

class RNNLM(BaseModel):
    """Standard implementation of a RNNLM. Model mainly consists of embeding layer,
    rnn layers(with dropout), and the full connection layer, which are all incuded
    in self.model_for_rnn
    """
    default_config = {
        "d_model": 512,       # the dim of model
        "rnn_type": 'lstm',   # the supported rnn type
        "num_layer": 2,       # the number of rnn layer
        "dropout_rate": 0.1,  # dropout for model
        "sos": -1,            # sos can be -1 or -2
        "eos": -1             # eos can be -1 or -2
    }
    def __init__(self, data_descriptions, config=None):
        """ config including the params for build lm """
        super(RNNLM, self).__init__()
        p = register_and_parse_hparams(self.default_config, config)
        self.num_class = (
            data_descriptions.num_class + 1
            if p.sos == p.eos
            else data_descriptions.num_class + 2
        )
        self.sos = self.num_class + p.sos
        self.eos = self.num_class + p.eos
        self.metric = tf.keras.metrics.Mean(name="AverageLoss")

        layers = tf.keras.layers
        input_features = layers.Input(
            shape=data_descriptions.sample_shape["output"],
            dtype=tf.int32
        )
        inner = tf.keras.layers.Embedding(self.num_class, p.d_model)(input_features)
        for _ in range(p.num_layer):
            inner = tf.keras.layers.Dropout(p.dropout_rate)(inner)
            inner = tf.keras.layers.RNN(
                cell=[SUPPORTED_RNNS[p.rnn_type](p.d_model)],
                return_sequences=True
            )(inner)
        inner = tf.keras.layers.Dropout(p.dropout_rate)(inner)
        inner = tf.keras.layers.Dense(self.num_class)(inner)
        self.rnnlm = tf.keras.Model(inputs=input_features, outputs=inner)

    def call(self, samples, training: bool = None):
        x = insert_sos_in_labels(samples['input'], self.sos)
        return self.rnnlm(x, training=training)

    def save_model(self, path):
        """
        for saving model and current weight, path is h5 file name, like 'my_model.h5'
        usage:
        new_model = tf.keras.models.load_model(path)
        """
        self.rnnlm.save(path)

    def get_loss(self, logits, samples, training=None):
        """ get loss """
        labels = samples['output']
        labels = insert_eos_in_labels(labels, self.eos, samples['output_length'])
        labels = tf.one_hot(labels, self.num_class)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        n_token = tf.cast(tf.reduce_sum(samples['output_length'] + 1), tf.float32)
        self.metric.update_state(loss)
        metrics = {self.metric.name: self.metric.result()}
        return tf.reduce_sum(loss) / n_token, metrics
