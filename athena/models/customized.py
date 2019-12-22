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
# Only support eager mode and TF>=2.0.0
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
""" a implementation of customized model """

import tensorflow as tf
from absl import logging
from athena import layers as athena_layers
from athena.utils.hparam import register_and_parse_hparams
from .base import BaseModel


def build_layer(input_data, layer_config):
    """
    Build one or multiple layers from layer configuration.
    args:
        layer_config: list of multiple layers, or dict of single layer parameters.
    returns: a Keras layer.
    """
    if isinstance(layer_config, list):
        # Recursively build each layer.
        output = input_data
        for one_layer_config in layer_config:
            logging.info(f"Layer conf: {one_layer_config}")
            output = build_layer(output, one_layer_config)
        return output

    # Build one layer.
    for layer_type in layer_config:
        layer_args = layer_config[layer_type]
        logging.info(f"Final layer config: {layer_type}: {layer_args}")
        layer_cls = getattr(athena_layers, layer_type)
        layer = layer_cls(**layer_args)
        return layer(input_data)


class CustomizedModel(BaseModel):
    """ a simple customized model """
    default_config = {
        "topo": [{}]
    }
    def __init__(self, num_classes, sample_shape, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config)

        logging.info(f"Network topology config: {self.hparams.topo}")
        input_feature = tf.keras.layers.Input(shape=sample_shape["input"], dtype=tf.float32)
        inner = build_layer(input_feature, self.hparams.topo)
        inner = tf.keras.layers.Dense(num_classes)(inner)
        self.model = tf.keras.Model(inputs=input_feature, outputs=inner)
        logging.info(self.model.summary())

    def call(self, samples, training=None):
        return self.model(samples["input"], training=training)
