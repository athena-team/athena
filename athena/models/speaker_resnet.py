# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Ne Luo
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
from .base import BaseModel
from ..loss import SoftmaxLoss, AMSoftmaxLoss, AAMSoftmaxLoss, ProtoLoss, AngleProtoLoss, GE2ELoss
from ..metrics import ClassificationAccuracy, EqualErrorRate
from ..layers.resnet_block import ResnetBasicBlock
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
    """ A sample implementation of resnet 34
        Reference to paper "Deep residual learning for image recognition"
        The implementation is the same as the standard resnet with 34 weighted layers,
        excepts using only 1/4 amount of filters to reduce computation.

        config:
            task: "speaker_identification" or "speaker_verification"
    """
    default_config = {
        "task": "speaker_identification",
        "hidden_size": 512,
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
        self.num_class = data_descriptions.num_class
        self.loss_function = self.init_loss(self.hparams.loss)
        self.task = self.hparams.task
        self.metric = ClassificationAccuracy(name="Top1-Accuracy")
        # equal error rate metric for evaluating
        if self.task == "speaker_verification":
            self.eval_metric = EqualErrorRate(name="EER")

        num_filters = self.hparams.num_filters
        num_layers = self.hparams.num_layers

        layers = tf.keras.layers
        input_features = layers.Input(shape=data_descriptions.sample_shape["input"],
                                      dtype=data_descriptions.sample_type["input"])

        inner = layers.Conv2D(filters=num_filters[0],
                              kernel_size=(7, 7),
                              strides=2,
                              padding="same")(input_features)
        inner = layers.BatchNormalization()(inner)
        inner = layers.MaxPool2D(pool_size=(3, 3),
                                 strides=2,
                                 padding="same")(inner)
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

        inner = layers.GlobalAveragePooling2D()(inner)
        inner = layers.Dense(self.hparams.hidden_size)(inner)
        self.speaker_resnet = tf.keras.Model(inputs=input_features,
                                             outputs=inner,
                                             name="speaker_resnet")
        print(self.speaker_resnet.summary())

    def call(self, samples, training=None):
        """ call model """
        if self.task == "speaker_verification" and "input_a" in samples and not training:
            input_features_a, input_features_b = samples["input_a"], samples["input_b"]
            output_a = self.speaker_resnet(input_features_a, training=training)
            output_b = self.speaker_resnet(input_features_b, training=training)
            output = [output_a, output_b]
        else:
            input_features = samples["input"]
            output = self.speaker_resnet(input_features, training=training)
        return output

    def init_loss(self, loss):
        """ initialize loss function """
        if loss == "softmax":
            loss_function = SUPPORTED_LOSS[loss](
                                embedding_size=self.hparams.hidden_size,
                                num_classes=self.num_class
                            )
        elif loss in ("amsoftmax", "aamsoftmax"):
            loss_function = SUPPORTED_LOSS[loss](
                                embedding_size=self.hparams.hidden_size,
                                num_classes=self.num_class,
                                m=self.hparams.margin,
                                s=self.hparams.scale
                            )
        elif loss in ("prototypical", "angular_prototypical", "ge2e"):
            # need special dataset builder
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return loss_function

    def get_loss(self, outputs, samples, training=None):
        # report eer rather than loss during evaluating for speaker verification
        if self.task == "speaker_verification" and "input_a" in samples and not training:
            loss, metrics = self.get_eer(outputs, samples, training)
        else:
            loss, outputs = self.loss_function(outputs, samples)
            self.metric.update_state(outputs, samples)
            metrics = {self.metric.name: self.metric.result()}
        return loss, metrics

    def get_eer(self, outputs, samples, training=False):
        """ get equal error rates """
        output_a, output_b = outputs
        cosine_simil = -tf.keras.losses.cosine_similarity(output_a, output_b, axis=1)
        self.eval_metric.update_state(cosine_simil, samples)
        metrics = {self.eval_metric.name: self.eval_metric.result()}
        # this is a fake loss, because for verification task which
        # training with classfication-based objectives, speakers in
        # dev set are not included in training set, so it is
        # impossible to calculate loss during evaluating
        loss = -1.0
        return loss, metrics

    def make_resnet_block_layer(self, num_filter, num_blocks, stride=1):
        """ returns sequential layer composed of resnet block  """
        resnet_block = tf.keras.Sequential()
        resnet_block.add(ResnetBasicBlock(num_filter, stride))

        for _ in range(1, num_blocks):
            resnet_block.add(ResnetBasicBlock(num_filter, 1))

        return resnet_block
