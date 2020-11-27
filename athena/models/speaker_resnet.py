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
""" an implementation of resnet model that can be used as a sample
    for speaker recognition """

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from .base import BaseModel
from ..loss import SoftmaxLoss, AMSoftmaxLoss, AAMSoftmaxLoss, ProtoLoss, AngleProtoLoss, GE2ELoss
from ..metrics import ClassificationAccuracy
from ..layers.resnet_block import residual_block
from ..utils.hparam import register_and_parse_hparams

SUPPORTED_LOSS = {
    "softmax": SoftmaxLoss,
    "amsoftmax": AMSoftmaxLoss,
    "aamsoftmax": AAMSoftmaxLoss,
    "prototypical": ProtoLoss,
    "angular_prototypical": AngleProtoLoss,
    "ge2e": GE2ELoss
}

class SpeakerResnet(BaseModel):
    """ A sample implementation of resnet 34 for speaker recognition
        Reference to paper "Deep residual learning for image recognition"
        The implementation is the same as the standard resnet with 34 weighted layers,
        excepts using only 1/4 amount of filters to reduce computation.
    """
    default_config = {
        "num_speakers": None,
        "hidden_size": None,
        "num_filters": [16, 32, 64, 128],
        "num_layers": [3, 4, 6, 3],
        "loss": "amsoftmax",
        "margin": 0.3,
        "scale": 15
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        # number of speakers
        self.num_class = self.hparams.num_speakers
        self.loss_function = self.init_loss(self.hparams.loss)
        self.metric = ClassificationAccuracy(top_k=1, name="Top1-Accuracy")
        num_filters = self.hparams.num_filters
        num_layers = self.hparams.num_layers

        bn_axis = 3
        weight_decay = 1e-4
        layers = tf.keras.layers
        input_features = layers.Input(shape=data_descriptions.sample_shape["input"],
                                      dtype=data_descriptions.sample_type["input"])
        inner = layers.Conv2D(
            filters=num_filters[0],
            kernel_size=7,
            strides=2, use_bias=False,
            kernel_initializer='orthogonal',
            kernel_regularizer=l2(weight_decay),
            padding='same',
            name='conv1_conv',
        )(input_features)
        inner = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(inner)
        inner = layers.Activation('relu', name='conv1_relu')(inner)
        inner = layers.MaxPooling2D(3, strides=2, name='pool1_pool', padding='same')(inner)

        inner = self.stack_block(inner, num_filters[0], num_layers[0], stride=1, name='conv2')
        inner = self.stack_block(inner, num_filters[1], num_layers[1], stride=2, name='conv3')
        inner = self.stack_block(inner, num_filters[2], num_layers[2], stride=2, name='conv4')
        inner = self.stack_block(inner, num_filters[3], num_layers[3], stride=2, name='conv5')
        self.embedding_size = num_filters[3]

        inner = layers.GlobalAveragePooling2D(name='avg_pool')(inner)

        if self.hparams.hidden_size is not None:
            self.embedding_size = self.hparams.hidden_size
            inner = layers.Dense(
                self.embedding_size,
                kernel_initializer='orthogonal',
                use_bias=True, trainable=True,
                kernel_regularizer=l2(weight_decay),
                bias_regularizer=l2(weight_decay),
                name='projection'
            )(inner)

        # create ResNet model
        self.speaker_resnet = tf.keras.Model(
            inputs=input_features, outputs=inner, name="resnet"
        )
        print(self.speaker_resnet.summary())

        embedding_input = layers.Input(
            shape=tf.TensorShape([None, self.embedding_size]), dtype=tf.float32
        )
        if self.hparams.loss == "softmax":
            output = layers.Dense(
                self.num_class,
                kernel_initializer='orthogonal',
                use_bias=True, trainable=True,
                kernel_regularizer=l2(weight_decay),
                bias_regularizer=l2(weight_decay),
                name='prediction'
            )(embedding_input)
        elif self.hparams.loss in ("amsoftmax", "aamsoftmax"):
            initializer = tf.initializers.GlorotNormal()
            weight = tf.Variable(
                initializer(shape=[self.embedding_size, self.num_class], dtype=tf.float32),
                trainable=True,
                name='prediction_w'
            )
            embedding_norm = tf.math.l2_normalize(embedding_input, axis=1)
            weight_norm = tf.math.l2_normalize(weight, axis=0)
            output = tf.matmul(embedding_norm, weight_norm, name='prediction_matmul')
        else:
            raise NotImplementedError

        # create final layer
        self.final_layer = tf.keras.Model(
            inputs=embedding_input, outputs=output, name="final_layer"
        )
        print(self.final_layer.summary())

    def stack_block(self, inner, filters, blocks, stride=2, name=None):
        """A set of stacked residual blocks."""
        inner = residual_block(inner, filters, stride=stride, name=name + '_block1')
        for i in range(2, blocks + 1):
            inner = residual_block(
                inner, filters, conv_shortcut=False, name=name + '_block' + str(i)
            )
        return inner

    def call(self, samples, training=None):
        """ call model """
        input_features = samples["input"]
        embedding = self.speaker_resnet(input_features, training=training)
        output = self.final_layer(embedding, training=training)
        return output

    def init_loss(self, loss):
        """initialize loss function

        Args:
            loss (str): "softmax" or "amsoftmax" or "aamsoftmax"

        Returns:
            loss function
        """
        if loss == "softmax":
            loss_function = SUPPORTED_LOSS[loss](
                                num_classes=self.num_class
                            )
        elif loss in ("amsoftmax", "aamsoftmax"):
            loss_function = SUPPORTED_LOSS[loss](
                                num_classes=self.num_class,
                                margin=self.hparams.margin,
                                scale=self.hparams.scale
                            )
        else:
            raise NotImplementedError
        return loss_function

    def get_loss(self, outputs, samples, training=None):
        loss = self.loss_function(outputs, samples)
        self.metric.update_state(outputs, samples)
        metrics = {self.metric.name: self.metric.result()}
        return loss, metrics

    def get_speaker_embedding(self, samples):
        """ call model """
        input_features = samples["input"]
        embedding = self.speaker_resnet(input_features, training=False)
        return embedding

    def decode(self, samples, hparams=None, decoder=None):
        outputs = self.call(samples, training=False)
        return outputs

    def verification(self, samples):
        """calculate cosine similarity between two segments

        Args:
            samples: a dict contains "input" and "input2"

        Returns:
            float: cosine similarity
        """
        input_features_a, input_features_b = samples["input"], samples["input2"]
        output_a = self.speaker_resnet(input_features_a, training=False)
        output_b = self.speaker_resnet(input_features_b, training=False)
        output_a_norm = tf.math.l2_normalize(output_a, axis=1)
        output_b_norm = tf.math.l2_normalize(output_b, axis=1)
        cosine_simil = tf.reduce_sum(output_a_norm * output_b_norm, axis=1)
        return cosine_simil
