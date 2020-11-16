# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Lekai Huang
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

""" Convolution module layer. """

from absl import logging
import tensorflow as tf
from .commons import ACTIVATIONS


class ConvModule(tf.keras.layers.Layer):
    """Convolution Module.

    See section 2.2 of this paper for a description of this technique:
    Conformer: Convolution-augmented Transformer for Speech Recognition (https://arxiv.org/abs/2005.08100)

    architecture:
        pointwise_conv(.)       # 2 * input_dim
           |
        glu(.)                  # input_dim
        depthwise_conv_1d(.)
        batch_norm(.)
        act(.)
           |
        pointwise_conv(.)
    """
    def __init__(self,
                 d_model,
                 kernel_size=32,
                 depth_multiplier=1,
                 activation='gelu'):
        super(ConvModule, self).__init__()

        self.pointwise_conv1 = tf.keras.layers.Conv1D(
            filters=2 * d_model,
            kernel_size=1, 
            strides=1,
            padding="valid", 
            name="pointwise_conv1"
        )
        self.glu = ACTIVATIONS['glu']

        self.depthwise_conv = tf.keras.layers.SeparableConv1D(
            filters=d_model, 
            kernel_size=kernel_size,
            strides=1,
            padding="same", 
            depth_multiplier=1, 
            name="depthwise_conv"
        )
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = ACTIVATIONS[activation] 
 
        self.pointwise_conv2 = tf.keras.layers.Conv1D(
            filters=d_model, 
            kernel_size=1, 
            strides=1,
            padding="valid", 
            name="pointwise_conv2"
        )

    def call(self, inputs, training=False):
        """Compute convolution module.

        Args:
            inputs : Input tensor (#batch, time, channels).

        Returns:
            tensor: Output tensor (#batch, time, channels).

        """
        # GLU mechanism
        outputs = self.pointwise_conv1(inputs, training=training) # (batch, time, 2*channel)
        outputs = self.glu(outputs) # (batch, time, channel)

        # 1D Depthwise Conv
        outputs = self.depthwise_conv(outputs, training=training) # (batch, time, channel)
        outputs = self.activation(self.bn(outputs, training=training)) # (batch, time, channel)

        outputs = self.pointwise_conv2(outputs, training=training) # (batch, time, channel)
        return outputs
