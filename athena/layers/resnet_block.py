# coding=utf-8
# Copyright (C) 2020 ATHENA AUTHORS; Ne Luo
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
# Only support eager mode and TF>=2.0.0
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes
"""an implementation of resnet block"""

import tensorflow as tf

class ResnetBasicBlock(tf.keras.layers.Layer):
    """Basic block of resnet
    Reference to paper "Deep residual learning for image recognition"
    """
    def __init__(self, num_filter, stride=1):
        super().__init__()
        layers = tf.keras.layers
        self.conv1 = layers.Conv2D(filters=num_filter,
                                   kernel_size=(3, 3),
                                   strides=stride,
                                   padding="same")
        self.conv2 = layers.Conv2D(filters=num_filter,
                                   kernel_size=(3, 3),
                                   strides=1,
                                   padding="same")
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.add = layers.add
        self.relu = tf.nn.relu
        self.downsample_layer = self.make_downsample_layer(
            num_filter=num_filter, stride=stride
        )

    def call(self, inputs):
        """call model"""
        output = self.conv1(inputs)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        residual = self.downsample_layer(inputs)
        output = self.add([residual, output])
        output = self.relu(output)
        return output

    def make_downsample_layer(self, num_filter, stride):
        """perform downsampling using conv layer with stride != 1"""
        if stride != 1:
            downsample = tf.keras.Sequential()
            downsample.add(tf.keras.layers.Conv2D(filters=num_filter,
                                                  kernel_size=(1, 1),
                                                  strides=stride))
            downsample.add(tf.keras.layers.BatchNormalization())
        else:
            downsample = lambda x: x
        return downsample
