# coding=utf-8
# Copyright (C) ATHENA AUTHORS; Xiangang Li; Shuaijiang Zhao; Yanguang Xu
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
""" audio dataset """
import os
import os.path
import sys
import kaldiio
import numpy as np
import tensorflow as tf
from absl import logging
from athena.data.datasets.base import SpeechBaseDatasetBuilder
from athena.data.datasets.preprocess import SpecAugment
from athena.transform import feats

class VoiceActivityDetectionDatasetKaldiIOBuilder(SpeechBaseDatasetBuilder):
    """VoiceActivityDetectionDatasetKaldiIOBuilder
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "left_context": 10,
        "right_context": 10,
        "num_cmvn_workers": 1,
        "cmvn_file": None,
        "speed_permutation": [1.0],
        "spectral_augmentation": None,
        "data_scps_dir": None,
        "apply_cmvn": True,
        "train_mode": 'normal_horovod',
        "feat_preprocess_type": "splice"
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        self.left_context = self.hparams.left_context
        self.right_context = self.hparams.right_context
        self.feat_preprocess_type = self.hparams.feat_preprocess_type
        if self.feat_preprocess_type == "splice":
            self.input_frame_num = self.left_context + 1 + self.right_context
        else:
            self.input_frame_num = 1
        if self.hparams.spectral_augmentation is not None:
            self.spectral_augmentation = SpecAugment(
                self.hparams.spectral_augmentation)
        if self.hparams.data_scps_dir == None:
            pass
            logging.warning("created a fake dataset")
        else:
            self.preprocess_data(self.hparams.data_scps_dir)

    def preprocess_data(self, data_scps_dir):
        """generate a list of tuples (wav_filename, wav_offset, wav_length_ms, transcript, label).
        """
        logging.info("Loading data from {}".format(data_scps_dir))
        self.kaldi_io_feats = kaldiio.load_scp(
            os.path.join(data_scps_dir, "feats.scp"))
        self.kaldi_io_labels = kaldiio.load_scp(
            os.path.join(data_scps_dir, "labels.scp"))
        
        # data checking
        if self.kaldi_io_feats.keys() != self.kaldi_io_labels.keys():
            logging.info("Error: feats.scp and labels.scp does not contain same keys, " +
                         "please check your data.")
            sys.exit()

        self.entries = []
        for key in self.kaldi_io_feats.keys():
            self.entries.append(key)

        return self

    def splice_feature(self, feature):
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
        feature_len = feature.shape[0]
        splice_feat = feature

        left_context_feat = feature
        for _ in range(self.left_context):
            left_context_feat = tf.concat(([feature[0]], left_context_feat[:-1]), axis=0)
            splice_feat = tf.concat((left_context_feat, splice_feat), axis=1)

        right_context_feat = feature
        for _ in range(self.right_context):
            right_context_feat = tf.concat((right_context_feat[1:], [feature[-1]]), axis=0)
            splice_feat = tf.concat((splice_feat, right_context_feat), axis=1)

        return splice_feat

    def __getitem__(self, index):
        """get a sample

        Args:
            index (int): index of the entries

        Returns:
            dict: sample::

            {
                "input": feat,
                "input_length": feat_length,
                "output_length": label_length,
                "output": label,
                "utt": utt
            }
        """
        key = self.entries[index]

        feat = self.kaldi_io_feats[key]
        feat = feat.reshape(feat.shape[0], feat.shape[1], 1)
        feat = tf.convert_to_tensor(feat)
        if self.hparams.apply_cmvn:
            feat = self.feature_normalizer(feat, 'global')
        if self.hparams.spectral_augmentation is not None:
            feat = tf.squeeze(feat).numpy()
            feat = self.spectral_augmentation(feat)
            feat = tf.expand_dims(tf.convert_to_tensor(feat), -1)
        if self.feat_preprocess_type == "splice":
            feat = self.splice_feature(feat)
        feat_length = feat.shape[0]

        label = self.kaldi_io_labels[key]
        label_length = len(label)

        # data checking
        if feat_length != label_length:
            logging.info("Error: the length of feat and label for {} is not same, ".format(key) +
                         "please check your data.")
            sys.exit()

        return {
            "input": feat,
            "input_length": feat_length,
            "output_length": label_length,
            "output": label,
            "utt": key 
        }

    @property
    def sample_type(self):
        """:obj:`@property`

        Returns:
            dict: sample_type of the dataset::

            {
                "input": tf.float32,
                "input_length": tf.int32,
                "output_length": tf.int32,
                "output": tf.int32,
            }
        """
        return {
            "input": tf.float32,
            "input_length": tf.int32,
            "output_length": tf.int32,
            "output": tf.int32,
            "utt": tf.string,
        }

    @property
    def sample_shape(self):
        """:obj:`@property`

        Returns:
            dict: sample_shape of the dataset::

            {
                "input": tf.TensorShape([None, dim, nc]),
                "input_length": tf.TensorShape([]),
                "output_length": tf.TensorShape([]),
                "output": tf.TensorShape([None]),
                "utt": tf.TensorShape([]),
            }
        """
        dim = self.audio_featurizer.dim * self.input_frame_num
        nc = self.audio_featurizer.num_channels
        return {
            "input": tf.TensorShape([None, dim, nc]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None]),
            "utt": tf.TensorShape([]),
        }

    @property
    def sample_signature(self):
        """:obj:`@property`

        Returns:
            dict: sample_signature of the dataset::

            {
                "input": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                utt": tf.TensorSpec(shape=(None), dtype=tf.string),
            }
        """
        dim = self.audio_featurizer.dim * self.input_frame_num
        nc = self.audio_featurizer.num_channels
        return (
            {
                "input": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "utt": tf.TensorSpec(shape=(None), dtype=tf.string),
            },
        )

