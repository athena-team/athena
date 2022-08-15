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
from typing import Tuple, Optional, List

import tensorflow as tf

from athena.layers.transformer import Transformer
from athena.layers.u2.transformer_u2_layer import TransformerU2EncoderLayer, TransformerU2DecoderLayer
from athena.utils.misc import add_optional_chunk_mask_u2, create_multihead_mask_u2


class TransformerU2(Transformer):
    """A transformer model. User is able to modify the attributes as needed.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu
            (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).

    Examples:
        >>> transformer_model = Transformer(nhead=16, num_encoder_layers=12)
        >>> src = tf.random.normal((10, 32, 512))
        >>> tgt = tf.random.normal((20, 32, 512))
        >>> out = transformer_model(src, tgt)
    """

    def __init__(
            self,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            unidirectional=False,
            look_ahead=0,
            custom_encoder=None,
            custom_decoder=None,
            conv_module_kernel_size=0,
            static_chunk_size: int = 0,
            use_dynamic_chunk: bool = False,
            use_dynamic_left_chunk: bool = False,
    ):
        super().__init__()
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layers = [
                TransformerU2EncoderLayer(
                    d_model, nhead, dim_feedforward,
                    dropout, activation, unidirectional, look_ahead,
                    conv_module_kernel_size=conv_module_kernel_size
                )
                for _ in range(num_encoder_layers)
            ]
            self.encoder = TransformerU2Encoder(encoder_layers, static_chunk_size,
                                                use_dynamic_chunk, use_dynamic_left_chunk)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layers = [
                TransformerU2DecoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation
                )
                for _ in range(num_decoder_layers)
            ]
            self.decoder = TransformerU2Decoder(decoder_layers)

        self.d_model = d_model
        self.nhead = nhead
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

    def call(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None,
             return_encoder_output=False, return_attention_weights=False, training=None):
        """Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            - src: :math:`(N, S, E)`.
            - tgt: :math:`(N, T, E)`.
            - src_mask: :math:`(N, S)`.
            - tgt_mask: :math:`(N, T)`.
            - memory_mask: :math:`(N, S)`.

            Note: [src/tgt/memory]_mask should be a ByteTensor where True values are positions
            that should be masked with float('-inf') and False values will be unchanged.
            This mask ensures that no information will be taken from position i if
            it is masked, and has a separate mask for each sequence in a batch.

            - output: :math:`(N, T, E)`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.

            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number

        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if src.shape[0] != tgt.shape[0]:
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.shape[2] != self.d_model or tgt.shape[2] != self.d_model:
            raise RuntimeError(
                "the feature number of src and tgt must be equal to d_model"
            )

        memory, encoder_mask = self.encoder(src, src_mask=src_mask, training=training)
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            return_attention_weights=return_attention_weights, training=training
        )
        if return_encoder_output:
            return output, memory
        return output


class TransformerU2Encoder(tf.keras.layers.Layer):
    """TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples:
        >>> encoder_layer = [TransformerU2EncoderLayer(d_model=512, nhead=8)
        >>>                    for _ in range(num_layers)]
        >>> transformer_encoder = TransformerU2Encoder(encoder_layer)
        >>> src = tf.random.gamma(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self,
                 encoder_layers,
                 static_chunk_size: int = 0,
                 use_dynamic_chunk: bool = False,
                 use_dynamic_left_chunk: bool = False,
                 ):
        super().__init__()
        self.layers = encoder_layers
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

    def call(self, src, src_mask=None, training=None):
        """Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
        """
        output = src
        for i in range(len(self.layers)):
            output, src_mask, fake_cnn_cache = self.layers[i](output, src_mask=src_mask, training=training)
        return output, src_mask

    def set_unidirectional(self, uni=False):
        """whether to apply trianglar masks to make transformer unidirectional
        """
        for layer in self.layers:
            layer.set_unidirectional(uni)


class TransformerU2Decoder(tf.keras.layers.Layer):
    """TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples:
        >>> decoder_layer = [TransformerDecoderLayer(d_model=512, nhead=8)
        >>>                     for _ in range(num_layers)]
        >>> transformer_decoder = TransformerDecoder(decoder_layer)
        >>> memory = tf.random.gamma(10, 32, 512)
        >>> tgt = tf.random.gamma(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, decoder_layers):
        super().__init__()
        self.layers = decoder_layers

    def call(self, tgt, memory, tgt_mask=None, memory_mask=None, return_attention_weights=False,
             training=None):
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).

        """
        output = tgt
        attention_weights = []

        for i in range(len(self.layers)):
            output, attention_weight = self.layers[i](
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                training=training,
            )
            attention_weights.append(attention_weight)
        if return_attention_weights:
            return output, attention_weights
        return output

