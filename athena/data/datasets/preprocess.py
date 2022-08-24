# coding=utf-8
# Copyright (C) ATHENA AUTHORS;
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

""" preprecessing for speech features

This code is modified from https://github.com/wenet-e2e/wenet.git.

"""
import random
from PIL import Image
from PIL.Image import BICUBIC
from ...utils.hparam import register_and_parse_hparams
import tensorflow as tf
import numpy as np

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
        "warp_for_time": False,
        "num_t_mask": 2,
        "num_f_mask": 2,
        "max_t": 50,
        "max_f": 10,
        "max_w": 80
    }

    def __init__(self, preprocess_config):
        hparams = register_and_parse_hparams(self.default_config, preprocess_config, cls=self.__class__)
        self.warp_for_time = hparams.warp_for_time
        self.num_t_mask = hparams.num_t_mask
        self.num_f_mask = hparams.num_f_mask
        self.max_t = hparams.max_t
        self.max_f = hparams.max_f
        self.max_w = hparams.max_w


    def __call__(self, feat):
        """spec augment preprocess for audio features

        Args:
            feat: audio features, shape should be [time_steps, dimension, channels]

        Returns:
            processed features
        """
        feat = self.spec_augmentation(feat)

        return feat

    def spec_augmentation(self, x):
        """ Deep copy x and do spec augmentation then return it

        Args:
            x: input feature, T * F 2D
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            max_w: max width of time warp

        Returns:
            augmented feature
        """
        y = np.copy(x)
        max_frames = y.shape[0]
        max_freq = y.shape[1]

        # time warp
        if self.warp_for_time and max_frames > self.max_w * 2:
            center = random.randrange(self.max_w, max_frames - self.max_w)
            warped = random.randrange(center - self.max_w, center + self.max_w) + 1

            left = Image.fromarray(x[:center]).resize((max_freq, warped), BICUBIC)
            right = Image.fromarray(x[center:]).resize(
                (max_freq, max_frames - warped), BICUBIC)
            y = np.concatenate((left, right), 0)
        # time mask
        for i in range(self.num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, self.max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(self.num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, self.max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        return y
