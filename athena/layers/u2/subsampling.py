# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Chaoyang Mei
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
# pylint: disable=too-few-public-methods, invalid-name
# pylint: disable=too-many-instance-attributes, no-self-use, too-many-arguments
from typing import Tuple, Any

import tensorflow as tf
from athena.layers.u2.embedding import PositionalEncodingU2


class BaseSubsampling(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> tf.Tensor:
        return self.pos_enc.position_encoding(offset, size)

    def compute_logit_length(self, input_length):
        """ used for get logit length """
        return input_length


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, num_filters: int, odim: int, feats_dim: int, dropout_rate: float,
                 pos_enc_class: tf.keras.layers.Layer):
        # def __init__(self, num_filters: int, odim: int):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=num_filters,
                                            kernel_size=(3, 3),
                                            strides=(2, 2),
                                            padding="valid",
                                            activation=tf.nn.relu6,
                                            use_bias=False,
                                            data_format="channels_last", )
        self.conv2 = tf.keras.layers.Conv2D(filters=num_filters,
                                            kernel_size=(3, 3),
                                            strides=(2, 2),
                                            padding="valid",
                                            activation=tf.nn.relu6,
                                            use_bias=False,
                                            data_format="channels_last", )
        self.out = tf.keras.layers.Dense(odim, activation=tf.nn.relu6)
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    # @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, 80, 1], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[None, None, None, None], dtype=tf.bool),
    #                               tf.TensorSpec(shape=None, dtype=tf.int32),)
    #              )
    def call(self,
             input: tf.Tensor,
             x_mask: tf.Tensor,
             offset: int = 0,
             ) -> Tuple[Any, Any, Any]:
        """Subsample x.

        Args:
            x (tf.Tensor): Input tensor (#batch, time, idim, 1).
            x_mask (tf.Tensor): Input mask (#batch, 1, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim, 1),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = self.conv1(input)
        x = self.conv2(x)  # (B, compute_logit_length(T), compute_logit_length(feats_dim), num_filters)
        _, _, dim, channels = x.get_shape().as_list()
        output_dim = dim * channels
        x = tf.keras.layers.Reshape((-1, output_dim))(x)
        x = self.out(x)

        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :, :-2:2][:, :, :, :-2:2]

    def compute_logit_length(self, input_length):
        """ used for get logit length """
        input_length = tf.cast(input_length, tf.float32)
        logit_length = tf.math.ceil(input_length / 2 - 1)
        logit_length = tf.math.ceil(logit_length / 2 - 1)
        logit_length = tf.cast(logit_length, tf.int32)
        return logit_length


if __name__ == "__main__":
    pos = PositionalEncodingU2(256)

    cnn2d_sub4 = Conv2dSubsampling4(80, 256, 0, pos)
    input_shape, mask_shape = (16, 460, 80, 1), (16, 1, 1, 460)
    # cnn2d_sub4.built(input_shape=input_shape)
    # cnn2d_sub4.trainable_variables
    input = tf.random.normal(input_shape, dtype=tf.float32)
    mask = tf.zeros(mask_shape)
    out_put, pos_emb, x_mask = cnn2d_sub4(input, mask)
    print(out_put.shape)
    print(x_mask.shape)
