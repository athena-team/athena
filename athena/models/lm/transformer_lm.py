# coding=utf-8
# Copyright (C) 2022 ATHENA AUTHORS; Chaoyang Mei
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
# pylint: disable=no-member, invalid-name
""" Transformer language model implementation"""

import tensorflow as tf
from athena.utils.misc import create_multihead_mask
from athena.utils.hparam import register_and_parse_hparams
from athena.layers.transformer import TransformerEncoderLayer, TransformerEncoder
from .nn_lm import NNLM


class TransformerLM(NNLM):
    """Standard implementation of a RNNLM. Model mainly consists of embeding layer,
    rnn layers(with dropout), and the full connection layer, which are all incuded
    in self.model_for_rnn
    """
    default_config = {
        "d_model": 512,       # the dim of model
        "num_layer": 2,       # the number of rnn layer
        "num_heads": 8,
        "dropout_rate": 0.1,  # dropout for model
        "sos": -1,            # sos can be -1 or -2
        "eos": -1             # eos can be -1 or -2
    }

    def __init__(self, data_descriptions, config=None):
        """ config including the params for build lm """
        super(TransformerLM, self).__init__()
        p = register_and_parse_hparams(self.default_config, config)
        self.num_class = (
            data_descriptions.num_class + 1
            if p.sos == p.eos
            else data_descriptions.num_class + 2
        )
        self.sos = self.num_class + p.sos
        self.eos = self.num_class + p.eos

        self.embeding = tf.keras.layers.Embedding(self.num_class, p.d_model)
        encoder_layer = [TransformerEncoderLayer(d_model=p.d_model, nhead=p.num_heads)
                         for _ in range(p.num_layer)]
        self.transformer_encoder = TransformerEncoder(encoder_layer)
        self.last_dropout = tf.keras.layers.Dropout(p.dropout_rate)
        self.last_layer = tf.keras.layers.Dense(self.num_class)

    def forward(self, inputs, input_lengths, training: bool = None):
        inner = self.embeding(inputs)
        inner_mask, _ = create_multihead_mask(inner, input_lengths, None, reverse=True)
        inner = self.transformer_encoder(inner, src_mask=inner_mask)
        inner = self.last_dropout(inner, training=training)
        output = self.last_layer(inner)
        return output

