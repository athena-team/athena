# coding=utf-8
# Copyright (C) 2022 ATHENA AUTHORS; Chaoyang Mei
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
""" NN language model interface"""

import tensorflow as tf
from ..base import BaseModel
from athena.utils.misc import insert_eos_in_labels, insert_sos_in_labels


class NNLM(BaseModel):
    """Standard implementation of a RNNLM. Model mainly consists of embeding layer,
    rnn layers(with dropout), and the full connection layer, which are all incuded
    in self.model_for_rnn
    """

    def __init__(self):
        """ config including the params for build lm """
        super(NNLM, self).__init__()
        self.metric = tf.keras.metrics.Mean(name="AverageLoss")

    def forward(self, inputs, inputs_length, training: bool = None):
        """
        do NN LM forward computation, for both train and decode.
        """
        raise NotImplementedError("NNLM.forward() haven't been implement!")

    def call(self, samples, training: bool = None):
        x = insert_sos_in_labels(samples['input'], self.sos)
        return self.forward(x, samples["input_length"]+1, training)

    def save_model(self, path):
        """
        for saving model and current weight, path is h5 file name, like 'my_model.h5'
        usage:
        new_model = tf.keras.models.load_model(path)
        """
        self.save(path)

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

    def get_ppl(self, loss):
        """loss is returned by self.get_loss(), and it is a tensor with none shape float value"""
        return tf.exp(loss).numpy()
