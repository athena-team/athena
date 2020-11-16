# coding=utf-8
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
# changed from the pytorch transformer implementation
# pylint: disable=invalid-name, too-many-instance-attributes
# pylint: disable=too-few-public-methods, too-many-arguments

""" the transformer model """
import tensorflow as tf
from .attention import MultiHeadAttention
from .conv_module import ConvModule
from .commons import ACTIVATIONS


class Transformer(tf.keras.layers.Layer):
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
        conv_module_kernel_size=0
    ):
        super().__init__()
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layers = [
                TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward,
                    dropout, activation, unidirectional, look_ahead,
                    conv_module_kernel_size=conv_module_kernel_size
                )
                for _ in range(num_encoder_layers)
            ]
            self.encoder = TransformerEncoder(encoder_layers)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layers = [
                TransformerDecoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation
                )
                for _ in range(num_decoder_layers)
            ]
            self.decoder = TransformerDecoder(decoder_layers)

        self.d_model = d_model
        self.nhead = nhead

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

        memory = self.encoder(src, src_mask=src_mask, training=training)
        output = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            return_attention_weights=return_attention_weights, training=training
        )
        if return_encoder_output:
            return output, memory
        return output


class TransformerEncoder(tf.keras.layers.Layer):
    """TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples:
        >>> encoder_layer = [TransformerEncoderLayer(d_model=512, nhead=8)
        >>>                    for _ in range(num_layers)]
        >>> transformer_encoder = TransformerEncoder(encoder_layer)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layers):
        super().__init__()
        self.layers = encoder_layers

    def call(self, src, src_mask=None, training=None):
        """Pass the input through the endocder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
        """
        output = src
        for i in range(len(self.layers)):
            output = self.layers[i](output, src_mask=src_mask, training=training)
        return output

    def set_unidirectional(self, uni=False):
        """whether to apply trianglar masks to make transformer unidirectional
        """
        for layer in self.layers:
            layer.set_unidirectional(uni)


class TransformerDecoder(tf.keras.layers.Layer):
    """TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples:
        >>> decoder_layer = [TransformerDecoderLayer(d_model=512, nhead=8)
        >>>                     for _ in range(num_layers)]
        >>> transformer_decoder = TransformerDecoder(decoder_layer)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
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


class TransformerEncoderLayer(tf.keras.layers.Layer):
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
            unidirectional=False, look_ahead=0, ffn=None, conv_module_kernel_size=0
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, unidirectional, look_ahead=look_ahead)
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

    def call(self, src, src_mask=None, training=None):
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            mask: the mask for the src sequence (optional).

        """
        out = self.self_attn(src, src, src, mask=src_mask)[0]
        out = self.norm1(src + self.dropout(out, training=training))
        if self.conv_module is not None:
            out = self.norm3(out + self.conv_module(out, training=training))
        out = self.norm2(out + self.ffn(out, training=training))

        return out

    def set_unidirectional(self, uni=False):
        """whether to apply trianglar masks to make transformer unidirectional
        """
        self.self_attn.attention.uni = uni


class TransformerDecoderLayer(tf.keras.layers.Layer):
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
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = tf.random(10, 32, 512)
        >>> tgt = tf.random(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.attn1 = MultiHeadAttention(d_model, nhead)
        self.attn2 = MultiHeadAttention(d_model, nhead)
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
