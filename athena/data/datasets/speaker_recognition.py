# coding=utf-8
# Copyright (C) 2020 ATHENA AUTHORS; Ne Luo
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
""" speaker dataset """

import os
from absl import logging
import tensorflow as tf
from athena.transform import AudioFeaturizer
from ...utils.hparam import register_and_parse_hparams
from ..feature_normalizer import FeatureNormalizer
from .base import BaseDatasetBuilder

class SpeakerRecognitionDatasetBuilder(BaseDatasetBuilder):
    """SpeakerRecognitionDatasetBuilder
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "cmvn_file": None,
        "cut_frame": None,
        "input_length_range": [20, 50000],
        "data_csv": None
    }

    def __init__(self, config=None):
        super().__init__()
        # hparams
        self.hparams = register_and_parse_hparams(
            self.default_config, config, cls=self.__class__)
        logging.info("hparams: {}".format(self.hparams))

        self.audio_featurizer = AudioFeaturizer(self.hparams.audio_config)
        self.feature_normalizer = FeatureNormalizer(self.hparams.cmvn_file)

        self.entries = []
        self.speakers = []

        if self.hparams.data_csv is not None:
            self.load_csv(self.hparams.data_csv)

    def reload_config(self, config):
        """reload the config
        """
        if config is not None:
            self.hparams.override_from_dict(config)

    def preprocess_data(self, data_csv_path):
        """generate a list of tuples
           (wav_filename, wav_length_ms, speaker_id, speaker_name).
        """
        logging.info("Loading data csv {}".format(data_csv_path))
        with open(data_csv_path, "r", encoding='utf-8') as data_csv:
            lines = data_csv.read().splitlines()[1:]
        lines = [line.split("\t", 3) for line in lines]
        lines.sort(key=lambda item: int(item[1]))
        self.entries = [tuple(line) for line in lines]
        self.speakers = list(set(line[-1] for line in lines))

        # apply input length filter
        self.filter_sample_by_input_length()
        return self

    def cut_features(self, feature):
        """cut acoustic featuers
        """
        length = self.hparams.cut_frame
        # randomly select a start frame
        max_start_frames = tf.shape(feature)[0] - length
        if max_start_frames <= 0:
            return feature
        start_frames = tf.random.uniform([], 0, max_start_frames, tf.int32)
        return feature[start_frames:start_frames + length, :, :]

    def load_csv(self, data_csv_path):
        """load csv file
        """
        return self.preprocess_data(data_csv_path)

    def __getitem__(self, index):
        """get a sample

        Args:
            index (int): index of the entries

        Returns:
            dict: sample::

            {
                "input": feat,
                "input_length": feat_length,
                "output_length": 1,
                "output": spkid
            }
        """
        audio_data, _, spkid, spkname = self.entries[index]
        feat = self.audio_featurizer(audio_data)
        feat = self.feature_normalizer(feat, spkname)
        if self.hparams.cut_frame is not None:
            feat = self.cut_features(feat)
        feat_length = feat.shape[0]
        spkid = [spkid]
        return {
            "input": feat,
            "input_length": feat_length,
            "output_length": 1,
            "output": spkid
        }

    def __len__(self):
        """return the number of data samples
        """
        return len(self.entries)

    @property
    def num_class(self):
        """:obj:`@property`

        Returns:
            int: the number of speakers
        """
        return len(self.speakers)

    @property
    def sample_type(self):
        """:obj:`@property`

        Returns:
            dict: sample_type of the dataset::

            {
                "input": tf.float32,
                "input_length": tf.int32,
                "output_length": tf.int32,
                "output": tf.int32
            }
        """
        return {
            "input": tf.float32,
            "input_length": tf.int32,
            "output_length": tf.int32,
            "output": tf.int32
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
                "output": tf.TensorShape([None])
            }
        """
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return {
            "input": tf.TensorShape([None, dim, nc]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None])
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
            },
        )

    def filter_sample_by_input_length(self):
        """filter samples by input length

        The length of filterd samples will be in [min_length, max_length)

        Returns:
            entries: a filtered list of tuples
            (wav_filename, wav_len, transcripts, speed, speaker)
        """
        min_len = self.hparams.input_length_range[0]
        max_len = self.hparams.input_length_range[1]
        filter_entries = []
        for items in self.entries:
            if int(items[1]) in range(min_len, max_len):
                filter_entries.append(items)
        self.entries = filter_entries
        return self

    def compute_cmvn_if_necessary(self, is_necessary=True):
        """compute cmvn file
        """
        if not is_necessary:
            return self
        if os.path.exists(self.hparams.cmvn_file):
            return self
        feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        with tf.device("/cpu:0"):
            self.feature_normalizer.compute_cmvn(
                self.entries, self.speakers, self.audio_featurizer, feature_dim
            )
        self.feature_normalizer.save_cmvn(["speaker", "mean", "var"])
        return self


class SpeakerVerificationDatasetBuilder(SpeakerRecognitionDatasetBuilder):
    """SpeakerVerificationDatasetBuilder
    """

    def __init__(self, config=None):
        super().__init__(config=config)

    def preprocess_data(self, data_csv_path):
        """generate a list of tuples
           (wav_filename_a, speaker_a, wav_filename_b, speaker_b, label).
        """
        logging.info("Loading data csv {}".format(data_csv_path))
        with open(data_csv_path, "r", encoding='utf-8') as data_csv:
            lines = data_csv.read().splitlines()[1:]
        lines = [line.split("\t", 4) for line in lines]
        self.entries = [tuple(line) for line in lines]
        return self

    def __getitem__(self, index):
        """get a sample

        Args:
            index (int): index of the entries

        Returns:
            dict: sample::

            {
                "input_a": feat_a,
                "input_b": feat_b,
                "output": [label]
            }
        """
        audio_data_a, speaker_a, audio_data_b, speaker_b, label = self.entries[index]
        feat_a = self.audio_featurizer(audio_data_a)
        feat_a = self.feature_normalizer(feat_a, speaker_a)
        feat_b = self.audio_featurizer(audio_data_b)
        feat_b = self.feature_normalizer(feat_b, speaker_b)
        return {
            "input_a": feat_a,
            "input_b": feat_b,
            "output": [label]
        }

    @property
    def sample_type(self):
        """:obj:`@property`

        Returns:
            dict: sample_type of the dataset::

            {
                "input_a": tf.float32,
                "input_b": tf.float32,
                "output": tf.int32
            }
        """
        return {
            "input_a": tf.float32,
            "input_b": tf.float32,
            "output": tf.int32
        }

    @property
    def sample_shape(self):
        """:obj:`@property`

        Returns:
            dict: sample_shape of the dataset::

            {
                "input_a": tf.TensorShape([None, dim, nc]),
                "input_b": tf.TensorShape([None, dim, nc]),
                "output": tf.TensorShape([None])
            }
        """
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return {
            "input_a": tf.TensorShape([None, dim, nc]),
            "input_b": tf.TensorShape([None, dim, nc]),
            "output": tf.TensorShape([None])
        }

    @property
    def sample_signature(self):
        """:obj:`@property`

        Returns:
            dict: sample_signature of the dataset::

            {
                "input_a": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "input_b":tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            }
        """
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return (
            {
                "input_a": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "input_b":tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            },
        )
