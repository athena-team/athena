# coding=utf-8
# Copyright (C) ATHENA AUTHORS
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
# pylint: disable=missing-function-docstring, invalid-name
""" misc """
import os
import wave
import tensorflow as tf
from absl import logging
import numpy as np


def mask_index_from_labels(labels, index):
    mask = tf.math.logical_not(tf.math.equal(labels, index))
    mask = tf.cast(mask, dtype=labels.dtype)
    return labels * mask


def insert_sos_in_labels(labels, sos):
    sos = tf.ones([tf.shape(labels)[0], 1], dtype=labels.dtype) * sos
    return tf.concat([sos, labels], axis=-1)


def remove_eos_in_labels(input_labels, labels_length):
    """remove eos in labels, batch size should be larger than 1
    assuming 0 as the padding and the last one is the eos
    """
    labels = input_labels[:, :-1]
    length = labels_length - 1
    max_length = tf.shape(labels)[1]
    mask = tf.sequence_mask(length, max_length, dtype=labels.dtype)
    labels = labels * mask
    labels.set_shape([None, None])
    return labels


def insert_eos_in_labels(input_labels, eos, labels_length):
    """insert eos in labels, batch size should be larger than 1
    assuming 0 as the padding,
    """
    zero = tf.zeros([tf.shape(input_labels)[0], 1], dtype=input_labels.dtype)
    labels = tf.concat([input_labels, zero], axis=-1)
    labels += tf.one_hot(labels_length, tf.shape(labels)[1], dtype=labels.dtype) * eos
    return labels


def generate_square_subsequent_mask(size):
    """Generate a square mask for the sequence. The masked positions are filled with float(1.0).
       Unmasked positions are filled with float(0.0).
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_multihead_mask(x, x_length, y, reverse=False):
    """Generate a square mask for the sequence for mult-head attention.
       The masked positions are filled with float(1.0).
       Unmasked positions are filled with float(0.0).
    """
    x_mask, y_mask = None, None
    if x is not None:
        x_mask = 1.0 - tf.sequence_mask(
            x_length, tf.shape(x)[1], dtype=tf.float32
        )
        x_mask = x_mask[:, tf.newaxis, tf.newaxis, :]
        if reverse:
            look_ahead_mask = generate_square_subsequent_mask(tf.shape(x)[1])
            x_mask = tf.maximum(x_mask, look_ahead_mask)
        x_mask.set_shape([None, None, None, None])
    if y is not None:
        y_mask = tf.cast(tf.math.equal(y, 0), tf.float32)
        y_mask = y_mask[:, tf.newaxis, tf.newaxis, :]
        if not reverse:
            look_ahead_mask = generate_square_subsequent_mask(tf.shape(y)[1])
            y_mask = tf.maximum(y_mask, look_ahead_mask)
        y_mask.set_shape([None, None, None, None])
    return x_mask, y_mask


def gated_linear_layer(inputs, gates, name=None):
    h1_glu = tf.keras.layers.multiply(inputs=[inputs, tf.sigmoid(gates)], name=name)
    return h1_glu


def validate_seqs(seqs, eos):
    """Discard end symbol and elements after end symbol

    Args:
        seqs: shape=(batch_size, seq_length)
        eos: eos id

    Returns:
        validated_preds: seqs without eos id
    """
    eos = tf.cast(eos, tf.int64)
    if tf.shape(seqs)[1] == 0:
        validated_preds = tf.zeros([tf.shape(seqs)[0], 1], dtype=tf.int64)
    else:
        if eos != 0:
            indexes = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
            a = tf.not_equal(eos, seqs)
            res = a[:, 0]
            indexes = indexes.write(0, res)
            for i in tf.range(1, tf.shape(a)[1]):
                res = tf.logical_and(res, a[:, i])
                indexes = indexes.write(i, res)
            res = tf.transpose(indexes.stack(), [1, 0])
            validated_preds = tf.where(tf.logical_not(res), tf.zeros_like(seqs), seqs)
        else:
            validated_preds = tf.where(tf.equal(eos, seqs), tf.zeros_like(seqs), seqs)
    validated_preds = tf.sparse.from_dense(validated_preds)
    counter = tf.cast(tf.shape(validated_preds.values)[0], tf.float32)
    return validated_preds, counter


def get_wave_file_length(wave_file):
    """get the wave file length(duration) in ms

    Args:
        wave_file: the path of wave file

    Returns:
        wav_length: the length(ms) of the wave file
    """
    if not os.path.exists(wave_file):
        logging.warning("Wave file {} does not exist!".format(wave_file))
        return 0
    with wave.open(wave_file) as wav_file:
        wav_frames = wav_file.getnframes()
        wav_frame_rate = wav_file.getframerate()
        wav_length = int(wav_frames / wav_frame_rate * 1000)  # get wave duration in ms
    return wav_length


def splice_numpy(x, context):
    """
    Splice a tensor along the last dimension with context.
    
    Example:

    >>> t = [[[1, 2, 3],
    >>>     [4, 5, 6],
    >>>     [7, 8, 9]]]
    >>> splice_tensor(t, [0, 1]) =
    >>>   [[[1, 2, 3, 4, 5, 6],
    >>>     [4, 5, 6, 7, 8, 9],
    >>>     [7, 8, 9, 7, 8, 9]]]

    Args:
        tensor: a tf.Tensor with shape (B, T, D) a.k.a. (N, H, W)
        context: a list of context offsets

    Returns:
        spliced tensor with shape (..., D * len(context))
    """
    # numpy can speed up 10%
    x = x.numpy()
    input_shape = np.shape(x)
    B, T = input_shape[0], input_shape[1]
    context_len = len(context)
    left_boundary = -1 * min(context) if min(context) < 0 else 0
    right_boundary = max(context) if max(context) > 0 else 0
    sample_range = ([0] * left_boundary + [i for i in range(T)] + [T - 1] * right_boundary)
    array = []
    for idx in range(context_len):
        pos = context[idx]
        if pos < 0:
            pos = len(sample_range) - T - max(context) + pos
        sliced = x[:, sample_range[pos : pos + T], :]
        array += [sliced]
    spliced = np.concatenate([i[:, :, np.newaxis, :] for i in array], axis=2)
    spliced = np.reshape(spliced, (B, T, -1))
    return tf.convert_to_tensor(spliced)

def set_default_summary_writer(summary_directory=None):
    if summary_directory is None:
        summary_directory = os.path.join(os.path.expanduser("~"), ".athena")
        summary_directory = os.path.join(summary_directory, "event")
    writer = tf.summary.create_file_writer(summary_directory)
    writer.set_as_default()

def tensor_shape(tensor):
    """Return a list with tensor shape. For each dimension,
       use tensor.get_shape() first. If not available, use tf.shape().
    """
    if tensor.get_shape().dims is None:
        return tf.shape(tensor)
    shape_value = tensor.get_shape().as_list()
    shape_tensor = tf.shape(tensor)
    ret = [shape_tensor[idx]
        if shape_value[idx] is None
        else shape_value[idx]
        for idx in range(len(shape_value))]
    return ret

def apply_label_smoothing(inputs, num_classes, smoothing_rate=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.
    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      num_classes: Number of classes.
      smoothing_rate: Smoothing rate.
    ```
    '''
    return ((1.0 - smoothing_rate) * inputs) + (smoothing_rate / num_classes)
