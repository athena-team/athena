# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Shuaijiang Zhao
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
from absl import logging
import tensorflow as tf
from athena.transform import AudioFeaturizer
from ...utils.hparam import register_and_parse_hparams
from ..feature_normalizer import FeatureNormalizer
from .base import BaseDatasetBuilder


class SpeechDatasetBuilder(BaseDatasetBuilder):
    """ SpeechDatasetBuilder

    Args:
        for __init__(self, config=None)

    Config:
        feature_config: the config file for feature extractor, default={'type':'Fbank'}
        data_csv: the path for original LDC HKUST,
            default='/tmp-data/dataset/opensource/hkust/train.csv'
        force_process: force process, if force_process=True, we will re-process the dataset,
            if False, we will process only if the out_path is empty. default=False

    Interfaces::
        __len__(self): return the number of data samples

        @property:
        sample_shape:
            {"input": tf.TensorShape([None, self.audio_featurizer.dim,
                                  self.audio_featurizer.num_channels]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None, self.audio_featurizer.dim *
                                  self.audio_featurizer.num_channels]),}
    """

    default_config = {
        "audio_config": {"type": "Fbank"},
        "cmvn_file": None,
        "input_length_range": [20, 50000]
    }

    def __init__(self, config=None):
        super().__init__()
        # hparams
        self.hparams = register_and_parse_hparams(
            self.default_config, config, cls=self.__class__)
        logging.info("hparams: {}".format(self.hparams))

        self.audio_featurizer = AudioFeaturizer(self.hparams.audio_config)
        self.feature_normalizer = FeatureNormalizer(self.hparams.cmvn_file)
        # self.file_path = self.hparams.data_csv
        # self.preprocess_data()

    def reload_config(self, config):
        """ reload the config """
        if config is not None:
            self.hparams.override_from_dict(config)

    def preprocess_data(self, file_path):
        """Generate a list of tuples (wav_filename, wav_length_ms, speaker)."""
        logging.info("Loading data from {}".format(file_path))
        with open(file_path, "r", encoding='utf-8') as file:
            lines = file.read().splitlines()
        headers = lines[0]
        lines = lines[1:]

        self.entries = []
        for line in lines:
            items = line.split("\t")
            wav_filename, length, speaker = items[0], items[1], 'global'
            if 'speaker' in headers.split("\t"):
                speaker = items[-1]
            self.entries.append(tuple([wav_filename, length, speaker]))
        self.entries.sort(key=lambda item: float(item[1]))

        self.speakers = []
        for _, _, speaker in self.entries:
            if speaker not in self.speakers:
                self.speakers.append(speaker)

        self.filter_sample_by_input_length()
        return self

    def load_csv(self, file_path):
        """ load csv file """
        return self.preprocess_data(file_path)

    def __getitem__(self, index):
        audio_file, _, speaker = self.entries[index]
        feat = self.audio_featurizer(audio_file)
        feat = self.feature_normalizer(feat, speaker)
        input_data = feat
        output_data = tf.reshape(
            feat, [-1, self.audio_featurizer.dim * self.audio_featurizer.num_channels]
        )

        return {
            "input": input_data,
            "input_length": input_data.shape[0],
            "output": output_data,
            "output_length": output_data.shape[0],
        }

    def __len__(self):
        """ return the number of data samples """
        return len(self.entries)

    @property
    def num_class(self):
        """ return the max_index of the vocabulary """
        target_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        return target_dim

    @property
    def speaker_list(self):
        """ return the speaker list """
        return self.speakers

    @property
    def audio_featurizer_func(self):
        """ return the audio_featurizer function """
        return self.audio_featurizer

    @property
    def sample_type(self):
        return {
            "input": tf.float32,
            "input_length": tf.int32,
            "output": tf.float32,
            "output_length": tf.int32,
        }

    @property
    def sample_shape(self):
        return {
            "input": tf.TensorShape(
                [None, self.audio_featurizer.dim, self.audio_featurizer.num_channels]
            ),
            "input_length": tf.TensorShape([]),
            "output": tf.TensorShape([None, None]),
            "output_length": tf.TensorShape([]),
        }

    @property
    def sample_signature(self):
        return (
            {
                "input": tf.TensorSpec(
                    shape=(None, None, None, None), dtype=tf.float32
                ),
                "input_length": tf.TensorSpec(shape=([None]), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                "output_length": tf.TensorSpec(shape=([None]), dtype=tf.int32),
            },
        )

    def filter_sample_by_input_length(self):
        """filter samples by input length

        The length of filterd samples will be in [min_length, max_length)

        Args:
            self.hparams.input_length_range = [min_len, max_len]
            min_len: the minimal length(ms)
            max_len: the maximal length(ms)
        returns:
            entries: a filtered list of tuples
            (wav_filename, wav_len, speaker)
        """
        min_len = self.hparams.input_length_range[0]
        max_len = self.hparams.input_length_range[1]
        filter_entries = []
        for wav_filename, wav_len, speaker in self.entries:
            if int(wav_len) in range(min_len, max_len):
                filter_entries.append(tuple([wav_filename, wav_len, speaker]))
        self.entries = filter_entries

    def compute_cmvn_if_necessary(self, is_necessary=True):
        """ compute cmvn file
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
        self.feature_normalizer.save_cmvn()
        return self
