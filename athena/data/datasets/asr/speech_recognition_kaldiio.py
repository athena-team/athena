# coding=utf-8
# Copyright (C) ATHENA AUTHORS; Xiangang Li; Shuaijiang Zhao; Ne Luo; Yanguang Xu; Jianwei Sun
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
import sys
import numpy as np
from absl import logging
import tensorflow as tf
import kaldiio
from athena.data.text_featurizer import TextFeaturizer
from athena.data.datasets.base import SpeechBaseDatasetBuilder
from athena.data.datasets.preprocess import SpecAugment


class SpeechRecognitionDatasetKaldiIOBuilder(SpeechBaseDatasetBuilder):
    """SpeechRecognitionDatasetKaldiIOBuilder
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "text_config": {"type": "vocab", "model": "athena/utils/vocabs/ch-en.vocab"},
        "cmvn_file": None,
        "remove_unk": True,
        "input_length_range": [20, 50000],
        "output_length_range": [1, 10000],
        "speed_permutation": [1.0],
        "spectral_augmentation": None,
        "data_csv": None,
        "data_scps_dir": None,
        "words": None,
        "apply_cmvn": True,
        "global_cmvn": True,
        "offline": False,
        "apply_sort_filter": True,
        "ignore_unk": False,
        "train_mode": 'normal_horovod',
        "rank_size": 1,
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        if self.hparams.data_scps_dir is None:
            pass
            logging.warning("created a fake dataset")
        self.text_featurizer = TextFeaturizer(self.hparams.text_config)
        if self.hparams.data_scps_dir is not None and \
                os.path.isfile(os.path.join(self.hparams.data_scps_dir, "feats.scp")) and \
                os.path.isfile(os.path.join(self.hparams.data_scps_dir, "labels.scp")):
            self.preprocess_kaldi_data(
                self.hparams.data_scps_dir, apply_sort_filter=self.hparams.apply_sort_filter)
        if self.hparams.spectral_augmentation is not None:
            self.spectral_augmentation = SpecAugment(
                self.hparams.spectral_augmentation)

    def preprocess_kaldi_data(self, file_dir, apply_sort_filter=True):
        """ Generate a list of tuples (feat_key, speaker). """
        logging.info("Loading kaldi-format feats.scp, labels.scp " +
                     "and utt2spk (optional) from {}".format(file_dir))
        self.kaldi_io_feats = kaldiio.load_scp(
            os.path.join(file_dir, "feats.scp"))
        self.kaldi_io_labels = kaldiio.load_scp(
            os.path.join(file_dir, "labels.scp"))

        # data checking
        if self.kaldi_io_feats.keys() != self.kaldi_io_labels.keys():
            logging.info("Error: feats.scp and labels.scp does not contain same keys, " +
                         "please check your data.")
            sys.exit()

        # initialize all speakers with 'global'
        # unless 'utterance_key speaker' is specified in "utt2spk"
        self.speakers = dict.fromkeys(self.kaldi_io_feats.keys(), 'global')
        if os.path.exists(os.path.join(file_dir, "utt2spk")):
            with open(os.path.join(file_dir, "utt2spk"), "r") as f:
                lines = f.readlines()
                for line in lines:
                    key, spk = line.strip().split(" ", 1)
                    self.speakers[key] = spk

        self.entries = []
        for key in self.kaldi_io_feats.keys():
            self.entries.append(tuple([key, self.speakers[key]]))

        if apply_sort_filter:
            logging.info(
                "Sorting and filtering data, this is very slow, please be patient ...")
            self.entries.sort(
                key=lambda item: self.kaldi_io_feats[item[0]].shape[0])
            unk = self.text_featurizer.unk_index
            if self.hparams.remove_unk and unk != -1:
                self.entries = list(filter(lambda x: unk not in
                                    self.kaldi_io_labels[x[0]], self.entries))
            # filter length of frames
            self.entries = list(filter(lambda x: self.kaldi_io_feats[x[0]].shape[0] in
                                range(self.hparams.input_length_range[0],
                                self.hparams.input_length_range[1]), self.entries))
            self.entries = list(filter(lambda x: self.kaldi_io_labels[x[0]].shape[0] in
                                range(self.hparams.output_length_range[0],
                                self.hparams.output_length_range[1]), self.entries))
        return self

    def __getitem__(self, index):
        key, speaker = self.entries[index]

        feat = self.kaldi_io_feats[key]
        feat = feat.reshape(feat.shape[0], feat.shape[1], 1)
        feat = tf.convert_to_tensor(feat)
        if self.hparams.global_cmvn:
           speaker = "global"
        if self.hparams.apply_cmvn:
            feat = self.feature_normalizer(feat, speaker)
        if self.hparams.spectral_augmentation is not None:
            feat = tf.squeeze(feat).numpy()
            feat = self.spectral_augmentation(feat)
            feat = tf.expand_dims(tf.convert_to_tensor(feat), -1)
        label = list(self.kaldi_io_labels[key])

        feat_length = feat.shape[0]
        label_length = len(label)
        utt = os.path.basename(key)
        return {
            "input": feat,
            "input_length": feat_length,
            "output_length": label_length,
            "output": label,
            "utt": utt
        }

    def compute_cmvn_if_necessary(self, is_necessary=True):
        """ compute cmvn file
        """
        if not is_necessary:
            return self
        if os.path.exists(self.hparams.cmvn_file):
            return self
        feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        with tf.device("/cpu:0"):
            self.feature_normalizer.compute_cmvn_kaldiio(
                self.entries, self.speakers, self.kaldi_io_feats, feature_dim
            )
        self.feature_normalizer.save_cmvn(["speaker", "mean", "var"])
        return self

    @property
    def num_class(self):
        """ return the max_index of the vocabulary + 1 """
        return len(self.text_featurizer)

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
            }
        """
        dim = self.audio_featurizer.dim
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
            }
        """
        dim = self.audio_featurizer.dim
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
