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
import csv
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


def subsequent_chunk_mask(size, chunk_size, num_left_chunks=-1):
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder
    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): size of history chunk
    Returns:
        torch.Tensor: mask
    Examples:
        >>> subsequent_mask(4, 2, 1)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 1, 1, 1],
         [0, 1, 1, 1]]
    """
    '''
    ret = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, 0:ending] = 1
        if history_chunk_size != -1:
            padding_zero_ending = max(0, (i // chunk_size)* chunk_size-history_chunk_size)
            ret[i, 0:padding_zero_ending] = 0.0
    return ret
    '''
    x = tf.broadcast_to(
        tf.cast(tf.range(size), dtype=tf.int32), (size, size))

    chunk_current = tf.math.floordiv(x, chunk_size) * chunk_size
    chunk_ending = tf.minimum(chunk_current + chunk_size, size)
    chunk_ending_ = tf.transpose(chunk_ending)
    ret = tf.math.greater(chunk_ending, chunk_ending_)
    if num_left_chunks >= 0:
        chunk_start = tf.maximum(
            0, tf.transpose(chunk_current) - num_left_chunks)
        history_mask = tf.math.less(x, chunk_start)
        # ret = tf.maximum(history_mask, ret)
        ret = tf.logical_or(history_mask, ret)

    return tf.cast(ret, dtype=tf.float32)


def add_optional_chunk_mask(xs: tf.Tensor, masks: tf.Tensor, max_len,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int, static_chunk_size: int,
                            num_decoding_left_chunks: int):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        # max_len = xs.shape[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            # chunk_size = np.random.randint(1, int(max_len), (1,)).item()
            chunk_size = tf.random.uniform(shape=[], minval=1, maxval=max_len, dtype=tf.int32)

            num_left_chunks = -1
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = tf.random.uniform(shape=[], minval=0, maxval=max_left_chunks, dtype=tf.int32)
        chunk_masks = subsequent_chunk_mask(max_len, chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = tf.expand_dims(chunk_masks, axis=0)  # (1, L, L)
        chunk_masks = tf.maximum(masks, chunk_masks)  # masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(max_len, static_chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = tf.expand_dims(chunk_masks, axis=0)  # (1, L, L)
        chunk_masks = tf.maximum(masks, chunk_masks)  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks


def generate_square_subsequent_mask_u2(size):
    """Generate a square mask for the sequence. The masked positions are filled with bool(True).
       Unmasked positions are filled with bool(False).
    """
    # mask = 1 - tf.linalg.band_part(tf.ones((size, size), dtype=tf.bool), -1, 0)
    mask = tf.logical_not(tf.linalg.band_part(tf.ones((size, size), dtype=tf.bool), -1, 0))
    return mask


def create_multihead_mask_u2(x, x_length, y, reverse=False):
    """Generate a square mask for the sequence for mult-head attention.
       The masked positions are filled with bool(True).
       Unmasked positions are filled with bool(False).
    """
    x_mask, y_mask = None, None
    if x is not None:
        x_mask = tf.logical_not(tf.sequence_mask(
            x_length, tf.shape(x)[1], dtype=tf.bool
        ))
        x_mask = x_mask[:, tf.newaxis, tf.newaxis, :]  # (B, 1, 1, L)
        if reverse:
            look_ahead_mask = generate_square_subsequent_mask_u2(tf.shape(x)[1])
            x_mask = tf.logical_or(x_mask, look_ahead_mask)
        x_mask.set_shape([None, None, None, None])
    if y is not None:
        y_mask = tf.math.equal(y, 0)
        y_mask = y_mask[:, tf.newaxis, tf.newaxis, :]
        if not reverse:
            look_ahead_mask = generate_square_subsequent_mask_u2(tf.shape(y)[1])
            # y_mask = tf.maximum(y_mask, look_ahead_mask)
            y_mask = tf.logical_or(y_mask, tf.cast(look_ahead_mask, tf.bool))
        y_mask.set_shape([None, None, None, None])
    return x_mask, y_mask


def subsequent_chunk_mask_u2(size, chunk_size, num_left_chunks=-1):
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder
    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): size of history chunk
    Returns:
        torch.Tensor: mask
    Examples:
        >>> subsequent_chunk_mask_u2(4, 2, 1)
        [[False, False, True, True],
         [False, False, True, True],
         [True, False, False, False],
         [True, False, False, False]]
    """
    '''
    ret = np.zeros((size, size), dtype=np.int32)
    for i in range(size):
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, 0:ending] = 1
        if history_chunk_size != -1:
            padding_zero_ending = max(0, (i // chunk_size)* chunk_size-history_chunk_size)
            ret[i, 0:padding_zero_ending] = 0.0
    return ret
    '''
    x = tf.broadcast_to(
        tf.cast(tf.range(size), dtype=tf.int32), (size, size))

    chunk_current = tf.math.floordiv(x, chunk_size) * chunk_size
    chunk_ending = tf.minimum(chunk_current + chunk_size, size)
    chunk_ending_ = tf.transpose(chunk_ending)
    ret = tf.math.greater(chunk_ending, chunk_ending_)
    if num_left_chunks >= 0:
        chunk_start = tf.maximum(
            0, tf.transpose(chunk_current) - num_left_chunks * chunk_size)
        history_mask = tf.math.less(x, chunk_start)
        # ret = tf.maximum(history_mask, ret)
        ret = tf.logical_or(history_mask, ret)

    return tf.cast(ret, dtype=tf.bool)


def add_optional_chunk_mask_u2(xs: tf.Tensor, masks: tf.Tensor, max_len,
                               use_dynamic_chunk: bool,
                               use_dynamic_left_chunk: bool,
                               decoding_chunk_size: int, static_chunk_size: int,
                               num_decoding_left_chunks: int):
    """ Apply optional mask for encoder.

    Args:
        xs (tf.Tensor): padded input, (B, L, D), L for max length
        mask (tf.Tensor): mask for xs, (B, 1, L) or (B, 1, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        # max_len = xs.shape[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            # chunk_size = np.random.randint(1, int(max_len), (1,)).item()
            chunk_size = tf.random.uniform(shape=[], minval=1, maxval=max_len, dtype=tf.int32)

            num_left_chunks = -1
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = tf.random.uniform(shape=[], minval=0, maxval=max_left_chunks, dtype=tf.int32)
        chunk_masks = subsequent_chunk_mask_u2(max_len, chunk_size,
                                               num_left_chunks)  # (L, L)
        chunk_masks = tf.expand_dims(chunk_masks, axis=0)  # (1, L, L)
        chunk_masks = tf.logical_or(masks, chunk_masks)  # masks & chunk_masks  # (B, L, L) or (B, 1, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask_u2(max_len, static_chunk_size,
                                               num_left_chunks)  # (L, L)
        chunk_masks = tf.expand_dims(chunk_masks, axis=0)  # (1, L, L)
        chunk_masks = tf.logical_or(masks, chunk_masks)  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks


def mask_finished_scores(scores, flag):
    """ for the score of finished hyps, mask the first to 0.0 and the others -inf 

    Args:
        scores: candidate scores at current step 
            shape: [batch_size*beam_size, beam_size]
        flag: end_flag, 1 means end
            shape: [batch_size*beam_size, 1]

    Returns:
        scores: masked scores for finished hyps
    """
    beam_size = tf.shape(scores)[-1]
    zero_mask = tf.zeros_like(flag, dtype=tf.bool)
    flag = tf.cast(flag, dtype=tf.bool)
    if beam_size > 1:
        mask_finished_others = tf.concat([zero_mask, tf.tile(flag, [1, beam_size - 1])], axis=1)
        mask_finished_first = tf.concat([flag, tf.tile(zero_mask, [1, beam_size - 1])], axis=1)
    else:
        mask_finished_others = zero_mask
        mask_finished_first = flag
    scores = tf.where(mask_finished_others, -float('inf'), scores)
    scores = tf.where(mask_finished_first, 0.0, scores)
    return scores


def mask_finished_preds(preds, flag, eos):
    """ for the finished hyps, mask all the selected word to eos

    Args:
        preds:
            shape:[batch_size*beam_size, beam_size]
        flag: end_flag 1 means end
            shape:[batch_size*beam_size, 1]

    Returns:
        preds: masked preds for finished hyps
    """
    beam_size = tf.shape(preds)[-1]
    flag = tf.cast(flag, dtype=tf.bool)
    mask_finished = tf.tile(flag, [1, beam_size])
    preds = tf.where(mask_finished, eos, preds)
    return preds


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
        indices = tf.not_equal(eos, seqs)
        validated_preds = tf.where(indices, seqs, 0)
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
    >>> splice_tensor(t, [0, 1])
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
        sliced = x[:, sample_range[pos: pos + T], :]
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


def get_dict_from_scp(vocab, func=lambda x: int(x[0])):
    unit2idx = {}
    with open(vocab, 'r', encoding='utf-8') as fid:
        for line in fid:
            parts = line.strip().split()
            if len(parts) >= 2:
                unit = parts[0]
                idx = func(parts[1:])
                unit2idx[unit] = idx
    return unit2idx


class athena_dialect(csv.Dialect):
    """Describe the usual properties of Excel-generated CSV files."""
    delimiter = '\t'
    quotechar = '"'
    doublequote = True
    skipinitialspace = True
    lineterminator = '\n'
    quoting = csv.QUOTE_MINIMAL


csv.register_dialect("athena_dialect", athena_dialect)


def read_csv_dict(csv_path):
    res = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, dialect="athena_dialect")
        for r in reader:
            res.append(r)
    return res


def read_csv(csv_path):
    csv_head = []
    csv_body = []
    csv_reader = csv.reader(open(csv_path), delimiter="\t")
    for i, row in enumerate(csv_reader):
        if i == 0:
            csv_head.append(row)
        else:
            csv_body.append(row)
    return csv_head, csv_body


def write_csv(csv_path, data: list):
    with open(csv_path, 'w') as csvfile:
        w = csv.DictWriter(csvfile, fieldnames=data[0].keys(), dialect="athena_dialect")
        w.writeheader()  # 写入列头
        for line in data:
            w.writerow(line)
