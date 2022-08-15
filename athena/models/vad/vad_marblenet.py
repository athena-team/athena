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

class TFReflectionPad1d(tf.keras.layers.Layer):
    """Tensorflow ReflectionPad1d module."""

    def __init__(self, padding_size, padding_type="CONSTANT", **kwargs):
        """Initialize TFReflectionPad1d module.

        Args:
            padding_size (int)
            padding_type (str) ("CONSTANT", "REFLECT", or "SYMMETRIC". Default is "REFLECT")
        """
        super().__init__(**kwargs)
        self.padding_size = padding_size
        self.padding_type = padding_type

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Padded tensor (B, T + 2 * padding_size, C).
        """
        return tf.pad(
            x,
            [[0, 0], [self.padding_size, self.padding_size], [0, 0]],
            self.padding_type,
        )

class BNResidualBlock(tf.keras.layers.Layer):
    """Tensorflow BN_Residual_Block module."""

    def __init__(
        self,
        repeat_times,
        kernel_size,
        filters,
        dilation_rate,
        strides,
        zoneout_rate,
        nonlinear_activation,
        nonlinear_activation_params,
        is_weight_norm,
        padding_type="REFLECT",
        **kwargs
    ):
        """Initialize BNResidualBlock module.
        Args:
            kernel_size (int): Kernel size.
            filters (int): Number of filters.
            dilation_rate (int): Dilation rate.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
        """
        super().__init__(**kwargs)
        self.conv_block = [
            TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate),
            tf.keras.layers.SeparableConv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                dilation_rate=dilation_rate,
            ),
            tf.keras.layers.BatchNormalization(),
        ]
        self.dropout_block = [
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            ),
            tf.keras.layers.Dropout(zoneout_rate),
        ]

        self.blocks = []
        for _ in range(repeat_times - 1):
            self.blocks.extend([
                TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate),
                tf.keras.layers.SeparableConv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    dilation_rate=dilation_rate,
                ),
                tf.keras.layers.BatchNormalization(),
                getattr(tf.keras.layers, nonlinear_activation)(
                    **nonlinear_activation_params
                ),
                tf.keras.layers.Dropout(zoneout_rate),
            ])
        self.blocks.extend([
            TFReflectionPad1d((kernel_size - 1) // 2 * dilation_rate),
            tf.keras.layers.SeparableConv1D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                dilation_rate=dilation_rate,
            ),
            tf.keras.layers.BatchNormalization(),
        ])

        self.shortcut = [
            tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=1,
                strides=1,
                name="shortcut",
            ),
            tf.keras.layers.BatchNormalization(),
        ]

        self.outputs = [
            getattr(tf.keras.layers, nonlinear_activation)(
                **nonlinear_activation_params
            ),
            tf.keras.layers.Dropout(zoneout_rate),
        ]

        # apply weightnorm
        if is_weight_norm:
            self._apply_weightnorm(self.blocks)
            self._apply_weightnorm(self.shortcut)
            self._apply_weightnomr(self.outputs)

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T, C).
        """
        _x = tf.identity(x)
        _s = tf.identity(x)
        for layer in self.blocks:
            _x = layer(_x)
        for layer in self.shortcut:
            _s = layer(_s)
        y = _s + _x
        for layer in self.outputs:
            y = layer(y)
        #return x + y
        return y

    def _apply_weightnorm(self, list_layers):
        """Try apply weightnorm for all layer in list_layers."""
        for i in range(len(list_layers)):
            try:
                layer_name = list_layers[i].name.lower()
                if "conv1d" in layer_name or "dense" in layer_name:
                    list_layers[i] = WeightNormalization(list_layers[i])
            except Exception:
                pass

class BNResidualBlocks(tf.keras.layers.Layer):
    default_config = {
            "kernel_size": [13, 15, 17],
            "filters": 128,
            "dilation_rate": 1,
            "stride": 1,
            "zoneout_rate": 0.0,
            "nonlinear_activation": "ReLU",
            "nonlinear_activation_params": {"threshold": 0.0},
            "sub_blocks": 2,
            "is_weight_norm": False
            }

    def __init__(self, config):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.blocks = []
        for index, kernel in enumerate(self.hparams.kernel_size):
            self.blocks.append(
                BNResidualBlock(
                    repeat_times = self.hparams.sub_blocks,
                    kernel_size = kernel,
                    filters = self.hparams.filters,
                    dilation_rate = self.hparams.dilation_rate,
                    strides = self.hparams.stride,
                    zoneout_rate = self.hparams.zoneout_rate,
                    nonlinear_activation = self.hparams.nonlinear_activation,
                    nonlinear_activation_params = self.hparams.nonlinear_activation_params,
                    is_weight_norm = self.hparams.is_weight_norm,
                    name="bn_residual_block_._{}".format(index),
                )
            )

    def call(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, T, C).
        Returns:
            Tensor: Output tensor (B, T, C).
        """
        _x = tf.identity(x)
        for layer in self.blocks:
            _x = layer(_x)
        return _x

class VadMarbleNet(BaseModel):
    '''
    implementation of a frame level or segment speech classification
    '''

    default_config = {
            "num_classes": 2,
            "encoder_output_channel": 1,
            "dn_block_conf":None,
            }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        layers = tf.keras.layers
        self.input_feats_dim = data_descriptions.audio_featurizer.dim
        self.input_frame_num = data_descriptions.input_frame_num
        self.input_feats_channels = data_descriptions.audio_featurizer.num_channels
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metric =  tf.keras.metrics.SparseCategoricalAccuracy()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.encoder_output_channel = self.hparams.encoder_output_channel
        self.decoder_input_dim = self.encoder_output_channel * self.input_frame_num

        # encoder architecture
        input_features = layers.Input(
                shape=data_descriptions.sample_shape["input"],
                dtype=tf.float32
                )
        inner = layers.Reshape((-1, self.input_feats_dim * self.input_feats_channels))(input_features)
        inner = TFReflectionPad1d((11 - 1) // 2 * 1)(inner)
        inner = layers.SeparableConv1D(
                filters=128,
                kernel_size=11,
                strides=1,
                dilation_rate=1)(inner)
        inner = BNResidualBlocks(self.hparams.dn_block_conf)(inner)
        inner = TFReflectionPad1d((29 - 1) // 2 * 2)(inner)
        inner = layers.SeparableConv1D(
                filters=128,
                kernel_size=29,
                strides=1,
                dilation_rate=2)(inner)
        inner = layers.SeparableConv1D(
                filters=128,
                kernel_size=1,
                strides=1,
                dilation_rate=1)(inner)
        inner = layers.Conv1D(
                filters=self.encoder_output_channel,
                kernel_size=1,
                strides=1,
                dilation_rate=1)(inner)

        # decoder architecture
        inner = layers.Reshape((1, self.decoder_input_dim))(inner)
        out = layers.Dense(self.hparams.num_classes)(inner)

        self.net = tf.keras.Model(inputs=input_features, outputs=out, name="vad_model")
        self.net.summary()

        self.tflite_model = self.build_model(data_descriptions)

    def call(self, samples, training=None):
        x = samples["input"]
        
        # expand the rank of input in testing phase, when batch_size is set to 1
        if len(x.get_shape().as_list()) == 3:
            x = tf.expand_dims(x, 0)
        x_len = tf.shape(x)[1]
        
        # reshape input tensor from [B, T, D, C] to [B * T, D, C]
        x = tf.reshape(x, (-1, self.input_frame_num, self.input_feats_dim * self.input_feats_channels))
        y = self.net(x, training=training)
        # reshape output tensor from [B * T, 1, num_class] to [B, T, num_class]
        y = tf.reshape(y, (-1, x_len, self.hparams.num_classes))
        return y

    def build_model(self, data_descriptions):
        input_features = tf.keras.layers.Input(
                shape= data_descriptions.sample_shape["input"],
                dtype=tf.float32,
                name="input_features")
        logits = self.net(input_features)
        prob = tf.nn.softmax(logits, name="softmax")
        tflite_model = tf.keras.Model(inputs=input_features, outputs=prob, name="vad")
        return tflite_model

    def get_loss(self, outputs, samples, training=None):
        loss = self.loss_function(samples["output"], outputs)
        self.metric.update_state(samples["output"], outputs)
        metrics = {self.metric.name: self.metric.result()}
        return loss, metrics
