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
""" an implementation of resnet model that can be used as a sample
    for speaker recognition """

import tensorflow as tf
from .base import BaseModel
from ..loss import SoftmaxLoss, MSELoss, AAMSoftmaxLoss, ProtoLoss, AngleProtoLoss, GE2ELoss
from ..metrics import MeanAbsoluteError
from ..layers.resnet_block import ResnetBasicBlock, stack_fn
from ..utils.hparam import register_and_parse_hparams
from tensorflow.keras.regularizers import l2

SUPPORTED_LOSS = {
    "softmax": SoftmaxLoss,
    "mseloss": MSELoss,
    "aamsoftmax": AAMSoftmaxLoss,
    "prototypical": ProtoLoss,
    "angular_prototypical": AngleProtoLoss,
    "ge2e": GE2ELoss
}

class AgeResnet(BaseModel):
    """ A sample implementation of resnet 34 for speaker recognition
        Reference to paper "Deep residual learning for image recognition"
        The implementation is the same as the standard resnet with 34 weighted layers,
        excepts using only 1/4 amount of filters to reduce computation.
    """
    default_config = {
        "num_speakers": None,
        "hidden_size": 128,
        "num_filters": [16, 32, 64, 128],
        "num_layers": [3, 4, 6, 3],
        "loss": "mseloss",
        "max_age": 100,
        "scale": 15
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        # maximum value of speakers' ages
        self.max_age = self.hparams.max_age

        self.loss_function = self.init_loss(self.hparams.loss)
        self.metric = MeanAbsoluteError(max_val=100, is_norm=False, name="MAE")

        num_filters = self.hparams.num_filters
        num_layers = self.hparams.num_layers

        layers = tf.keras.layers
        input_features = layers.Input(shape=data_descriptions.sample_shape["input"],
                                      dtype=data_descriptions.sample_type["input"])

        bn_axis = 3
        weight_decay = 1e-4

        inner = layers.Conv2D(
            16, 7,
            strides=2, use_bias=False,
            kernel_initializer='orthogonal',
            kernel_regularizer=l2(weight_decay),
            padding='same',
            name='conv1_conv',
        )(input_features)

        inner = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(inner)
        inner = layers.Activation('relu', name='conv1_relu')(inner)

        inner = layers.MaxPooling2D(3, strides=2, name='pool1_pool', padding='same')(inner)

        inner = self.make_resnet_block_layer(num_filter=num_filters[0],
                                             num_blocks=num_layers[0],
                                             stride=1)(inner)
        inner = self.make_resnet_block_layer(num_filter=num_filters[1],
                                             num_blocks=num_layers[1],
                                             stride=2)(inner)
        inner = self.make_resnet_block_layer(num_filter=num_filters[2],
                                             num_blocks=num_layers[2],
                                             stride=2)(inner)
        inner = self.make_resnet_block_layer(num_filter=num_filters[3],
                                             num_blocks=num_layers[3],
                                             stride=2)(inner)

        inner = layers.GlobalAveragePooling2D(name='avg_pool')(inner)

        inner = layers.Dense(self.hparams.hidden_size,
                           kernel_initializer='orthogonal',
                           use_bias=True, trainable=True,
                           kernel_regularizer=l2(weight_decay),
                           bias_regularizer=l2(weight_decay),
                           name='projection')(inner)

        inner = layers.Dense(1,
                            activation=tf.keras.activations.relu,
                            kernel_initializer='orthogonal',
                            use_bias=False, trainable=True,
                            kernel_regularizer=l2(weight_decay),
                            bias_regularizer=l2(weight_decay),
                            name='prediction')(inner)
        inner = tf.keras.activations.relu(inner, max_value=self.max_age)
        # Create model
        self.age_resnet = tf.keras.Model(inputs=input_features, outputs=inner, name="resnet-34")
        print(self.age_resnet.summary())

    def make_resnet_block_layer(self, num_filter, num_blocks, stride=1):
        """ returns sequential layer composed of resnet block """
        resnet_block = tf.keras.Sequential()
        resnet_block.add(ResnetBasicBlock(num_filter, stride))
        for _ in range(1, num_blocks):
            resnet_block.add(ResnetBasicBlock(num_filter, 1))
        return resnet_block

    def call(self, samples, training=None):
        """ call model """
        input_features = samples["input"]
        output = self.age_resnet(input_features, training=training)
        return output

    def init_loss(self, loss):
        """ initialize loss function """
        if loss == "softmax":
            loss_function = SUPPORTED_LOSS[loss](
                                num_classes=self.num_class
                            )
        elif loss == "mseloss":
            loss_function = SUPPORTED_LOSS[loss](
                                max_scale=self.max_age,
                                is_norm=False
                            )
        else:
            raise NotImplementedError
        return loss_function

    def get_loss(self, outputs, samples, training=None):
        loss = self.loss_function(outputs, samples)
        self.metric.update_state(outputs, samples)
        metrics = {self.metric.name: self.metric.result()}
        return loss, metrics


    def decode(self, samples, hparams=None, decoder=None):
        outputs = self.call(samples, training=False)
        return outputs

