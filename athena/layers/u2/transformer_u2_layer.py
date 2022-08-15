# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Chaoyang Mei
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
# changed from the pytorch transformer implementation
# pylint: disable=invalid-name, too-many-instance-attributes
# pylint: disable=too-few-public-methods, too-many-arguments

""" the transformer model """
from typing import Optional

import tensorflow as tf
from athena.layers.conv_module import ConvModule
from athena.layers.commons import ACTIVATIONS
from athena.layers.u2.attention_u2 import MultiHeadAttentionU2


class TransformerU2EncoderLayer(tf.keras.layers.Layer):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples:
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = tf.random(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
            self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu",
            unidirectional=False, look_ahead=0, ffn=None, conv_module_kernel_size=0, concat_after: bool = False,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttentionU2(d_model, nhead, unidirectional, look_ahead=look_ahead)
        # Implementation of Feedforward model
        layers = tf.keras.layers
        if ffn is None:
            self.ffn = tf.keras.Sequential(
                [
                    layers.Dense(
                        dim_feedforward,
                        activation=ACTIVATIONS[activation],
                        kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                            stddev=0.02
                        ),
                        input_shape=(None, d_model,),
                    ),
                    layers.Dropout(dropout, input_shape=(dim_feedforward,)),
                    layers.Dense(
                        d_model,
                        kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                            stddev=0.02
                        ),
                        input_shape=(None, dim_feedforward,),
                    ),
                    layers.Dropout(dropout, input_shape=(d_model,)),
                ]
            )
        else:
            self.ffn = ffn

        self.norm1 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.norm2 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.dropout = layers.Dropout(dropout, input_shape=(d_model,))

        self.conv_module = None
        self.norm3 = None
        if conv_module_kernel_size > 0:
            self.conv_module = ConvModule(d_model, kernel_size=conv_module_kernel_size, activation=activation)
            self.norm3 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.concat_after = concat_after

    # @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, None, 1], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[1, 0, 256], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[1, 0, 256], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[1, 0, 256], dtype=tf.float32),
    #                               tf.TensorSpec(shape=None, dtype=tf.bool),]
    #              )
    def call(self,
             src: tf.Tensor,
             src_mask: Optional[tf.Tensor] = None,
             output_cache: Optional[tf.Tensor] = None,
             cnn_cache: Optional[tf.Tensor] = None,
             training: Optional[bool] = None, ):
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask:  tf.zeros([1, 0, 256], dtype=tf.float32) the mask for the src sequence (optional).

        """
        # print("Tracing with transformer_u2_layer 102", "src", src, "src_mask:", src_mask,
        #           "output_cache:", output_cache, "cnn_cache:", cnn_cache)
        residual = src
        if output_cache is None or tf.shape(output_cache)[1] < tf.constant(1):
            src_q = src
        else:
            # assert tf.shape(output_cache)[0] == tf.shape(src)[0]
            # assert output_cache.shape[2] == self.size
            # assert tf.shape(output_cache)[1] < tf.shape(src)[1]
            chunk = tf.shape(src)[1] - tf.shape(output_cache)[1]
            src_q = src[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            src_mask = src_mask[:, -chunk:, :]

        out = self.self_attn(src, src, src_q, mask=src_mask)[0]
        out = self.norm1(residual + self.dropout(out, training=training))
        if self.conv_module is not None:
            out = self.norm3(out + self.conv_module(out, training=training))
        out = self.norm2(out + self.ffn(out, training=training))

        if not (output_cache is None or tf.shape(output_cache)[1] < tf.constant(1)):
            out = tf.concat([output_cache, out], axis=1)

        fake_cnn_cache = cnn_cache
        return out, src_mask, fake_cnn_cache

    def set_unidirectional(self, uni=False):
        """whether to apply trianglar masks to make transformer unidirectional
        """
        self.self_attn.attention.uni = uni


class TransformerU2DecoderLayer(tf.keras.layers.Layer):
    """TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    Reference:
        "Attention Is All You Need".

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples:
        >>> decoder_layer = TransformerU2DecoderLayer(d_model=512, nhead=8)
        >>> memory = tf.random(10, 32, 512)
        >>> tgt = tf.random(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(
            self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.attn1 = MultiHeadAttentionU2(d_model, nhead)
        self.attn2 = MultiHeadAttentionU2(d_model, nhead)
        # Implementation of Feedforward model
        layers = tf.keras.layers
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(
                    dim_feedforward,
                    activation=ACTIVATIONS[activation],
                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                        stddev=0.02
                    ),
                    input_shape=(d_model,)
                ),
                layers.Dropout(dropout, input_shape=(dim_feedforward,)),
                layers.Dense(
                    d_model,
                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                        stddev=0.02
                    ),
                    input_shape=(dim_feedforward,)
                ),
                layers.Dropout(dropout, input_shape=(d_model,)),
            ]
        )

        self.norm1 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.norm2 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.norm3 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.dropout1 = layers.Dropout(dropout, input_shape=(d_model,))
        self.dropout2 = layers.Dropout(dropout, input_shape=(d_model,))

    def call(self, tgt, memory, tgt_mask=None, memory_mask=None, training=None):
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
        """
        out = self.attn1(tgt, tgt, tgt, mask=tgt_mask)[0]
        out = self.norm1(tgt + self.dropout1(out, training=training))
        out2, decoder_weights = self.attn2(memory, memory, out, mask=memory_mask)
        out = self.norm2(out + self.dropout2(out2, training=training))
        out = self.norm3(out + self.ffn(out, training=training))
        return out, decoder_weights
