# coding=utf-8
# Copyright (C) ATHENA AUTHORS; Shuaijiang Zhao
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
from .commons import ACTIVATIONS
from .conv_module import ConvModule

class ConformerCTC(tf.keras.layers.Layer):
    """A conformer CTC model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017.
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
    Examples::
        >>> conformer_model = Conformer(nhead=16, num_encoder_layers=12)
        >>> src = tf.random.normal((10, 32, 512))
        >>> tgt = tf.random.normal((20, 32, 512))
        >>> out = transformer_model(src, tgt)
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        cnn_module_kernel=15,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        custom_encoder=None,
    ):
        super().__init__()
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layers = [
                ConformerEncoderLayer(
                    d_model, nhead, cnn_module_kernel, dim_feedforward, dropout, activation
                )
                for _ in range(num_encoder_layers)
            ]
            self.encoder = ConformerEncoder(encoder_layers)

        self.d_model = d_model
        self.nhead = nhead

    def call(self, src, src_mask=None, return_encoder_output=False, training=None):
        """Take in and process masked source sequences.
        Args:
            src: the sequence to the encoder (required).
            src_mask: the additive mask for the src sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
        Shape:
            - src: :math:`(N, S, E)`.
            - src_mask: :math:`(N, S)`.
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
            >>> output = conformer_ctc_model(src, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if src.shape[2] != self.d_model:
            raise RuntimeError(
                "the feature number of src must be equal to d_model"
            )

        memory = self.encoder(src, src_mask=src_mask, training=training)

        return memory


class ConformerEncoder(tf.keras.layers.Layer):
    """ConformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = [ConformerEncoderLayer(d_model=512, nhead=8)
        >>>                    for _ in range(num_layers)]
        >>> transformer_encoder = ConformerEncoder(encoder_layer)
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
        Shape:
            see the docs in Transformer class.
        """
        output = src
        for i in range(len(self.layers)):
            output = self.layers[i](output, src_mask=src_mask, training=training)
        return output


class ConformerEncoderLayer(tf.keras.layers.Layer):
    """ConformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> src = tf.random(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self, d_model, nhead, cnn_module_kernel=15, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
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

        self.conv_module = ConvModule(d_model, kernel_size=cnn_module_kernel)
        self.norm1 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.norm2 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.norm3 = layers.LayerNormalization(epsilon=1e-8, input_shape=(d_model,))
        self.dropout = layers.Dropout(dropout, input_shape=(d_model,))

    def call(self, src, src_mask=None, training=None):
        """Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            mask: the mask for the src sequence (optional).
        Shape:
            see the docs in Transformer class.
        """
        out = src + 1/2 * self.ffn(src, training=training)
        out = out + self.self_attn(out, out, out, mask=src_mask)[0]
        out = self.norm1(out + self.dropout(out, training=training))
        out = out + self.norm3(out + self.conv_module(out, training=training))
        out = self.norm2(out + 1/2 * self.ffn1(out, training=training))

        return out
