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
import tensorflow_addons as tfa
from athena.layers.functional import splice
from athena.utils.misc import gated_linear_layer


class PositionalEncoding(tf.keras.layers.Layer):
    """positional encoding can be used in transformer"""

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


class ScaledPositionalEncoding(PositionalEncoding):
    """scaled positional encoding,
       reference: https://arxiv.org/pdf/1809.08895.pdf"""
    def __init__(self, d_model, max_position=800):
        super().__init__(d_model, max_position, scale=False)

    def build(self, _):
        self.alpha = self.add_weight(
            name="alpha", initializer=tf.keras.initializers.constant(1)
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x += self.alpha * self.pos_encoding[:, :seq_len, :]
        return x


class Collapse4D(tf.keras.layers.Layer):
    """collapse4d can be used in cnn-lstm for speech processing
       reshape from [N T D C] -> [N T D*C]
    """

    def call(self, x):
        return collapse4d(x)


class Gelu(tf.keras.layers.Layer):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU. Original paper: https://arxiv.org/abs/1606.08415

    Args:
        x: float Tensor to perform activation.
    Returns:
        x: with the GELU activation applied.
    """

    def call(self, x):
        return gelu(x)


class TdnnLayer(tf.keras.layers.Layer):
    """An implementation of Tdnn Layer
    Args:
        context: a int of left and right context, or a list of context indexes, e.g. (-2, 0, 2).
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


class DownSampleBlock(tf.keras.layers.Layer):
    """conv2d downsample block for stargan, instance norm is used because batch size is 1
    """
    def __init__(self, filters, kernel_size, strides):
        super(DownSampleBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.norm1 = tfa.layers.InstanceNormalization(epsilon=1e-8)
        self.norm2 = tfa.layers.InstanceNormalization(epsilon=1e-8)

    def call(self, x):
        h1 = self.conv1(x)
        h1_norm = self.norm1(h1)
        h1_gates = self.conv2(x)
        h1_gates_norm = self.norm2(h1_gates)
        h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_gates_norm)
        return h1_glu


class UpSampleBlock(tf.keras.layers.Layer):
    """conv2d upsample block for stargan, instance norm is used because batch size is 1
    """
    def __init__(self, filters, kernel_size, strides):
        super(UpSampleBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.norm1 = tfa.layers.InstanceNormalization(epsilon=1e-8)
        self.norm2 = tfa.layers.InstanceNormalization(epsilon=1e-8)

    def call(self, x):
        h1 = self.conv1(x)
        h1_norm = self.norm1(h1)
        h1_gates = self.conv2(x)
        h1_gates_norm = self.norm2(h1_gates)
        h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_gates_norm)
        return h1_glu


class ZoneOutCell(tf.keras.layers.LSTMCell):
    """Wrapper for LSTM cell to create ZoneOut Cell

    inspired by:
    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_tensorflow.py
    Published by one of 'https://arxiv.org/pdf/1606.01305.pdf' paper writers.
    """

    def __init__(self, zoneout_rate=0., **kwargs):
        super().__init__(**kwargs)
        self.zoneout_rate = zoneout_rate
        self.drop_layer = tf.keras.layers.Dropout(self.zoneout_rate)

    def call(self, inputs, states, training: bool = None):
        """Runs vanilla LSTM Cell and applies zoneout.
        """
        # Apply vanilla LSTM
        outputs, new_states = super().call(inputs, states, training=training)
        if self.zoneout_rate == 0:
            return outputs, new_states
        # Apply zoneout
        h = (1 - self.zoneout_rate) * \
            self.drop_layer(new_states[0] - states[0], training=training) + \
            states[0]
        c = (1 - self.zoneout_rate) * \
            self.drop_layer(new_states[1] - states[1], training=training) + \
            states[1]
        return outputs, [h, c]

    def get_config(self):
        config = super().get_config()
        config['zoneout_rate'] = self.zoneout_rate
        return config


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
