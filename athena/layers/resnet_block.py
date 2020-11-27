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
""" an implementation of resnet block """

import tensorflow as tf
from tensorflow.keras.regularizers import l2

def residual_block(inner, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    bn_axis = 3
    weight_decay = 1e-4
    layers = tf.keras.layers

    if conv_shortcut:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride,
            use_bias=False,
            kernel_initializer='orthogonal',
            kernel_regularizer=l2(weight_decay),
            name=name + '_0_conv'
        )(inner)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = inner

    inner = layers.Conv2D(
        filters, kernel_size,
        strides=stride, padding='same',
        use_bias=False,
        kernel_initializer='orthogonal',
        kernel_regularizer=l2(weight_decay),
        name=name + '_1_conv'
    )(inner)
    inner = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(inner)
    inner = layers.Activation('relu', name=name + '_1_relu')(inner)

    inner = layers.Conv2D(
        filters, kernel_size,
        strides=1, padding='same',
        use_bias=False,
        kernel_initializer='orthogonal',
        kernel_regularizer=l2(weight_decay),
        name=name + '_2_conv'
    )(inner)
    inner = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(inner)
    inner = layers.Add(name=name + '_add')([shortcut, inner])
    inner = layers.Activation('relu', name=name + '_out')(inner)
    return inner
