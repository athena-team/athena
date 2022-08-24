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
from athena.layers.u2.conv_module_u2 import ConvModuleU2


class ConformerU2EncoderLayer(tf.keras.layers.Layer):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = ConformerU2EncoderLayer(d_model=512, nhead=8)
        >>> src = tf.random(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self, d_model, nhead, cnn_module_kernel=15, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = MultiHeadAttentionU2(d_model, nhead)
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
                    input_shape=(d_model,),
                ),
                layers.Dropout(dropout, input_shape=(dim_feedforward,)),
                layers.Dense(
                    d_model,
                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                        stddev=0.02
                    ),
                    input_shape=(dim_feedforward,),
                ),
                layers.Dropout(dropout, input_shape=(d_model,)),
            ]
        )

        self.ffn1 = tf.keras.Sequential(
            [
                layers.Dense(
                    dim_feedforward,
                    activation=ACTIVATIONS[activation],
                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                        stddev=0.02
                    ),
                    input_shape=(d_model,),
                ),
                layers.Dropout(dropout, input_shape=(dim_feedforward,)),
                layers.Dense(
                    d_model,
                    kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                        stddev=0.02
                    ),
                    input_shape=(dim_feedforward,),
                ),
                layers.Dropout(dropout, input_shape=(d_model,)),
            ]
        )

        self.conv_module = ConvModuleU2(d_model, kernel_size=cnn_module_kernel)
        self.norm1 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.norm2 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.norm3 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.dropout = layers.Dropout(dropout, input_shape=(d_model,))

    # @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, None, 1], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[1, 0, 256], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[1, 0, 256], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[1, 0, 256], dtype=tf.float32),
    #                               tf.TensorSpec(shape=None, dtype=tf.bool),]
    #              )
    def call(self,
             src: tf.Tensor,
             src_mask: Optional[tf.Tensor] = None,
             mask_pad: Optional[tf.Tensor] = None,
             output_cache: Optional[tf.Tensor] = None,
             cnn_cache: Optional[tf.Tensor] = None,
             training: Optional[bool] = None, ):
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask:  tf.zeros([1, 0, 256], dtype=tf.float32) the mask for the src sequence (optional).

        """
        # src = self.norm_ff_macaron(src)
        src = src + 1 / 2 * self.ffn(src, training=training)

        residual = src
        # src = self.norm_ff_macaron(src)

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

        out = residual + self.self_attn(src, src, src_q, mask=src_mask)[0]
        out = self.norm1(out + self.dropout(out, training=training))

        residual = out
        out, new_cnn_cache = self.conv_module(out, mask_pad, cnn_cache, training=training)
        out = residual + self.norm3(residual + out)
        out = self.norm2(out + 1/2 * self.ffn1(out, training=training))

        if output_cache is not None:
            out = tf.concat((output_cache, out), axis=1)

        return out, src_mask, new_cnn_cache


class ConformerU2DecoderLayer(tf.keras.layers.Layer):
    """ConformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> decoder_layer = ConformerU2DecoderLayer(d_model=512, nhead=8)
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
