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
# pylint: disable=invalid-name
"""Utils for common layers."""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from ..utils.misc import tensor_shape


def make_positional_encoding(position, d_model):
    """generate a postional encoding list"""

    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def collapse4d(x, name=None):
    """reshape from [N T D C] -> [N T D*C]
    using tf.shape(x), which generate a tensor instead of x.shape
    """
    with ops.name_scope(name, "collapse4d") as scope:
        shape = tensor_shape(x)
        N = shape[0]
        T = shape[1]
        D = shape[2]
        C = shape[3]
        DC = D * C
        out = tf.reshape(x, [N, T, DC])
        return out


def splice(x, context):
    """
    Splice a tensor along the last dimension with context.

    Example:

    >>> t = [[[1, 2, 3],
    >>>       [4, 5, 6],
    >>>       [7, 8, 9]]]
    >>> splice_tensor(t, [0, 1]) =
    >>>     [[[1, 2, 3, 4, 5, 6],
    >>>     [4, 5, 6, 7, 8, 9],
    >>>     [7, 8, 9, 7, 8, 9]]]

    Args:
        tensor: a tf.Tensor with shape (B, T, D) a.k.a. (N, H, W)
        context: a list of context offsets

    Returns:
        spliced tensor with shape (..., D * len(context))
    """
    input_shape = tf.shape(x)
    B, T = input_shape[0], input_shape[1]
    context_len = len(context)
    array = tf.TensorArray(x.dtype, size=context_len)
    for idx, offset in enumerate(context):
        begin = offset
        end = T + offset
        if begin < 0:
            begin = 0
            sliced = x[:, begin:end, :]
            tiled = tf.tile(x[:, 0:1, :], [1, abs(offset), 1])
            final = tf.concat((tiled, sliced), axis=1)
        else:
            end = T
            sliced = x[:, begin:end, :]
            tiled = tf.tile(x[:, -1:, :], [1, abs(offset), 1])
            final = tf.concat((sliced, tiled), axis=1)
        array = array.write(idx, final)
    spliced = array.stack()
    spliced = tf.transpose(spliced, (1, 2, 0, 3))
    spliced = tf.reshape(spliced, (B, T, -1))
    return spliced


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def glu(x, axis=-1):
    """Gated Linear Unit.

    Original paper: https://arxiv.org/abs/1612.08083

    Args:
      x: float Tensor to perform activation.
    Returns:
      `x` with the GLU activation applied.
    """
    a, b = tf.split(x, 2, axis=axis)
    b = tf.nn.sigmoid(b)
    return tf.multiply(a, b)
