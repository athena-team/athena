# -*- coding: utf-8 -*-
# Copyright (C) ATHENA AUTHORS; Yanguang Xu; Jianwei Sun
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
# pylint: disable=no-member, invalid-name
import struct
import numpy as np
import random
import logging
import kaldiio
from absl import logging
import os
import sys
import tensorflow as tf
from ..base import BaseDatasetBuilder
from ..preprocess import SpecAugment

class SpeechWakeupDatasetKaldiIOBuilder(BaseDatasetBuilder):
    """
    Dataset builder for RNN model. The builder mix the spliced frame in one dim
    For example (1, 1323)
    The input data format is (batch, t, dim, channel)
    For example (b, t, 1323, 1)
    The output data format is (batch, timestep)
    """

    default_config = {
            "data_dir":None,
            "left_context":10,
            "right_context":10,
            "feat_dim":63,
            "spectral_augmentation": None
            } 

    def __init__(self, config=None):
        super().__init__(config=config)
        self.preprocess_data(self.hparams.data_dir)
        if self.hparams.spectral_augmentation is not None:
            self.spectral_augmentation = SpecAugment(
                self.hparams.spectral_augmentation)

    def preprocess_data(self, data_dir=""):
        self.kaldi_io_feats = kaldiio.load_scp(os.path.join(data_dir, "feats.scp"))
        self.kaldi_io_labels = kaldiio.load_scp(os.path.join(data_dir, "labels.scp"))
        '''
        if self.kaldi_io_feats.keys() != self.kaldi_io_labels.keys():
            logging.info("Error: feats.scp and labels.scp does not contain same keys, " + \
                    "please check your data.")
            sys.exit()
        '''
        self.entries = []
        for key in self.kaldi_io_feats.keys():
            self.entries.append(key)

    def __getitem__(self, index):
        key = self.entries[index]
        feat = self.kaldi_io_feats[key]
        feat = feat.reshape(feat.shape[0], feat.shape[1], 1)
        feat = tf.convert_to_tensor(feat)
        if self.hparams.spectral_augmentation is not None:
            feat = tf.squeeze(feat).numpy()
            feat = self.spectral_augmentation(feat)
            feat = tf.expand_dims(tf.convert_to_tensor(feat), -1)
        label = int(self.kaldi_io_labels[key])
        feat_length = feat.shape[0]
        label_length = 1
        return {
                "input": feat,
                "input_length":feat_length,
                "output_length":label_length,
                "output":label
                }

    def splice_feature(self, feature, input_left_context, input_right_context):
        """ splice features according to input_left_context and input_right_context
        input_left_context: the left features to be spliced,
           repeat the first frame in case out the range
        input_right_context: the right features to be spliced,
           repeat the last frame in case out the range
        Args:
            feature: the input features, shape may be [timestamp, dim, 1]
        returns:
            splice_feat: the spliced features
        """
        splice_feat = feature

        left_context_feat = feature
        for _ in range(input_left_context):
            left_context_feat = tf.concat(([feature[0]], left_context_feat[:-1]), axis=0)
            splice_feat = tf.concat((left_context_feat, splice_feat), axis=1)

        right_context_feat = feature
        for _ in range(input_right_context):
            right_context_feat = tf.concat((right_context_feat[1:], [feature[-1]]), axis=0)
            splice_feat = tf.concat((splice_feat, right_context_feat), axis=1)

        return splice_feat

    @property
    def sample_type(self):
        return {
                "input": tf.float32,
                "input_length": tf.int32,
                "output_length": tf.int32,
                "output": tf.int32,
                }

    @property
    def sample_shape(self):
        dim = self.hparams.feat_dim
        return {
                "input": tf.TensorShape([None, dim, 1]), 
                "input_length": tf.TensorShape([]),
                "output_length": tf.TensorShape([]),
                "output": tf.TensorShape([]),
                }

    @property
    def sample_signature(self):
        dim = self.hparams.feat_dim
        return ({
            "input": tf.TensorSpec(shape=(None, None, dim, 1), dtype=tf.float32),
            "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
            "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
            "output": tf.TensorSpec(shape=(None),dtype=tf.int32 ),
            },)



