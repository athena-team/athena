# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li
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
# pylint: disable=no-self-use, missing-function-docstring
"""Utils for common layers."""

import tensorflow as tf
from athena.layers.functional import make_positional_encoding, collapse4d, gelu

from athena.layers.functional import splice


class PositionalEncoding(tf.keras.layers.Layer):
    """ positional encoding can be used in transformer """

    def __init__(self, d_model, max_position=800, scale=False):
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        self.pos_encoding = make_positional_encoding(max_position, d_model)

    def call(self, x):
        """ call function """
        seq_len = tf.shape(x)[1]
        if self.scale:
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        return x


class Collapse4D(tf.keras.layers.Layer):
    """ callapse4d can be used in cnn-lstm for speech processing
    reshape from [N T D C] -> [N T D*C]
    """

    def call(self, x):
        return collapse4d(x)


class Gelu(tf.keras.layers.Layer):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """

    def call(self, x):
        return gelu(x)


class TdnnLayer(tf.keras.layers.Layer):
    """ An implement of Tdnn Layer
    Args:
        context: a int of left and right context, or
        a list of context indexes, e.g. (-2, 0, 2).
        output_dim: the dim of the linear transform
    """

    def __init__(self, context, output_dim, use_bias=False, **kwargs):
        super().__init__(**kwargs)

        if hasattr(context, "__iter__"):
            self.context_size = len(context)
            self.context_list = context
        else:
            self.context_size = context * 2 + 1
            self.context_list = range(-context, context + 1)

        self.output_dim = output_dim
        self.linear = tf.keras.layers.Dense(output_dim, use_bias=use_bias)

    def call(self, x, training=None, mask=None):
        x = splice(x, self.context_list)
        x = self.linear(x, training=training, mask=mask)
        return x


SUPPORTED_RNNS = {
    "lstm": tf.keras.layers.LSTMCell,
    "gru": tf.keras.layers.GRUCell,
    "cudnnlstm": tf.keras.layers.LSTMCell,
    "cudnngru": tf.keras.layers.GRUCell
}


ACTIVATIONS = {
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "elu": tf.nn.elu,
    "selu": tf.nn.selu,
    "gelu": gelu,
    "leaky_relu": tf.nn.leaky_relu,
    "sigmoid": tf.nn.sigmoid,
    "softplus": tf.nn.softplus,
    "softsign": tf.nn.softsign,
    "tanh": tf.nn.tanh,
}
