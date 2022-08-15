# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Chaoyang Mei
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
# pylint: disable=too-few-public-methods, invalid-name
# pylint: disable=too-many-instance-attributes, no-self-use, too-many-arguments
import tensorflow as tf

from athena import make_positional_encoding


class PositionalEncodingU2(tf.keras.layers.Layer):
    """positional encoding can be used in transformer U2"""

    def __init__(self, d_model, max_position=5000, dropout_rate=0.0, scale=False):
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        if self.scale:
            self.xscale = tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        self.pos_encoding = make_positional_encoding(max_position, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x,
             offset: int = 0):
        """ call function """
        # print("Tracing with embedding 22", x, offset)
        seq_len = tf.shape(x)[1]
        pos_emb = self.pos_encoding[:, offset:offset + seq_len, :]
        if self.scale:
            x *= self.xscale
        x += self.pos_encoding[:,  offset:offset + seq_len, :]
        return self.dropout(x), pos_emb
