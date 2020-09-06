# coding=utf-8
# Copyright (C) ATHENA AUTHORS; Ruixiong Zhang;
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

""" preprecessing for speech features """
import random
import tensorflow as tf
from PIL import Image
from ...utils.hparam import register_and_parse_hparams
import tensorflow as tf

class SpecAugment:
    """Implementation of specaugument from paper "SpecAugment: A Simple Data
    Augmentation Method for Automatic Speech Recognition"

    Args:
        preprocess_config: it contains configs below::

            time_warping: warped time parameter, should be in (0, time / 2),
                a random horizontal center point in (W, time_steps - W) will be warped
                either to left or right by a distance chosen from range [0, W) randomly.
            time_masking: masked time range, should be in (0, time_steps),
                the final masked time steps will be [t_0, t_0 + t),
                t is random from[0, T), t_0 is random from [0, time_steps - t)
            frequency_masking: masked frequency range, should be in (0, dimension),
                the final masked frequencies will be [f_0, f_0 + f),
                f is random from[0, F), f_0 is random from [0, dimension - f)
            mask_cols: masking operation is executed mask_cols times in each axis
    """

    default_config = {
        "time_warping": 0,
        "time_masking": 0,
        "frequency_masking": 0,
        "mask_cols": 0,
        "mask_type": "mean"
    }

    def __init__(self, preprocess_config):
        hparams = register_and_parse_hparams(self.default_config, preprocess_config, cls=self.__class__)
        self.time_warping = hparams.time_warping
        self.time_masking = hparams.time_masking
        self.frequency_masking = hparams.frequency_masking
        self.mask_cols = hparams.mask_cols
        self.mask_type = hparams.mask_type

    def __call__(self, feat):
        """spec augment preprocess for audio features

        Args:
            feat: audio features, shape should be [time_steps, dimension, channels]

        Returns:
            processed features
        """
        feat = self.feat_time_warping(feat)
        feat = self.feat_masking(feat, axis=0, mask_num=self.time_masking)
        feat = self.feat_masking(feat, axis=1, mask_num=self.frequency_masking)
        return feat

    def feat_time_warping(self, feat):
        """time warping for spec agument

        Args:
            feat: audio features, shape should be [time_steps, dimension, channels]

        Returns:
            time warped features
        """
        time_steps = feat.shape[0]
        if self.time_warping >= time_steps - self.time_warping - 1 or self.time_warping == 0:
            return feat
        feat = tf.squeeze(feat).numpy()
        center = random.randrange(self.time_warping + 1, time_steps - self.time_warping)
        distance = random.randrange(-self.time_warping + 1, self.time_warping)
        warped = center + distance
        left = Image.fromarray(feat[:center]).resize(
            (feat.shape[1], warped), Image.BICUBIC)
        right = Image.fromarray(feat[center:]).resize(
          (feat.shape[1], time_steps - warped), Image.BICUBIC)
        feat[:warped] = left
        feat[warped:] = right
        warped_feat = tf.expand_dims(tf.convert_to_tensor(feat), -1)
        return warped_feat

    def feat_masking(self, feat, axis=0, mask_num=0):
        """masking for spec augment

        Args:
            feat: audio features, shape should be [time_steps, dimension, channels]
            axis (int, optional): the axis to be masked. Defaults to 0.
            mask_num (int, optional): masked time or frequency range. Defaults to 0.

        Returns:
            masked features
        """
        time_steps, freq_dim, channels = feat.shape

        dim_size = tf.shape(feat)[axis]
        if mask_num > dim_size or mask_num == 0 or self.mask_cols == 0:
            return feat

        total_t = tf.random.uniform([self.mask_cols, 1], 0, mask_num, tf.int32)
        max_t = tf.reduce_max(total_t)
        t_0 = tf.random.uniform([self.mask_cols, 1], 0, dim_size - max_t, tf.int32)
        t_end = t_0 + total_t
        t_0 = tf.tile(t_0, [1, dim_size]) # (mask_cols, dim_size)
        t_end = tf.tile(t_end, [1, dim_size]) # (mask_cols, dim_size)
        base_mask = tf.expand_dims(tf.range(dim_size), axis=0)
        base_mask = tf.tile(base_mask, [self.mask_cols, 1])
        # (time_mask_cols, time_steps)
        mask = tf.math.logical_xor(t_end <= base_mask, base_mask < t_0)
        final_mask = mask[0]
        for mask_bran in mask:
            final_mask = tf.math.logical_and(final_mask, mask_bran)
        if axis == 0:
            final_mask = tf.tile(final_mask[:, tf.newaxis, tf.newaxis], [1, freq_dim, channels])
        elif axis == 1:
            final_mask = tf.tile(final_mask[tf.newaxis, :, tf.newaxis], [time_steps, 1, channels])
        if self.mask_type == "mean":
            mask_metrics = tf.tile(tf.reduce_mean(feat, keepdims=True),
                                   [time_steps, freq_dim, channels])
        else:
            mask_metrics = tf.zeros(shape=[time_steps, freq_dim, channels])
        feat = tf.where(final_mask, feat, mask_metrics)
        return feat
