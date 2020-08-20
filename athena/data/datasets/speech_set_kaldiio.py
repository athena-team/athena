# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Shuaijiang Zhao; Ne Luo
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
import kaldiio
from athena.transform import AudioFeaturizer
from ...utils.hparam import register_and_parse_hparams
from ..feature_normalizer import FeatureNormalizer
from .base import BaseDatasetBuilder


class SpeechDatasetKaldiIOBuilder(BaseDatasetBuilder):
    """SpeechDatasetKaldiIOBuilder
    """

    default_config = {
        "audio_config": {"type": "Fbank"},
        "cmvn_file": None,
        "input_length_range": [20, 50000],
        "data_csv": None,
        "data_scps_dir": None
    }

    def __init__(self, config=None):
        super().__init__()
        # hparams
        self.hparams = register_and_parse_hparams(
            self.default_config, config, cls=self.__class__)
        logging.info("hparams: {}".format(self.hparams))

        self.audio_featurizer = AudioFeaturizer(self.hparams.audio_config)
        self.feature_normalizer = FeatureNormalizer(self.hparams.cmvn_file)

        if self.hparams.data_scps_dir is not None:
            self.load_scps(self.hparams.data_scps_dir)

    def reload_config(self, config):
        """ reload the config """
        if config is not None:
            self.hparams.override_from_dict(config)

    def preprocess_data(self, file_dir, apply_sort_filter=True):
        """ generate a list of tuples (feat_key, speaker). """
        logging.info("Loading kaldi-format feats.scp and utt2spk (optional) from {}".format(file_dir))
        self.kaldi_io_feats = kaldiio.load_scp(os.path.join(file_dir, "feats.scp"))

        # initialize all speakers with 'global' unless 'utterance_key speaker' is specified in "utt2spk"
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
            logging.info("Sorting and filtering data, this is very slow, please be patient ...")
            self.entries.sort(key=lambda item: self.kaldi_io_feats[item[0]].shape[0])
            self.filter_sample_by_input_length()
        return self

    def load_scps(self, file_dir):
        """ load kaldi-format feats.scp and utt2spk (optional) """
        return self.preprocess_data(file_dir)

    def __getitem__(self, index):
        key, speaker = self.entries[index]
        feat = self.kaldi_io_feats[key]
        feat = feat.reshape(feat.shape[0], feat.shape[1], 1)
        feat = tf.convert_to_tensor(feat)
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

        Returns:
            entries: a filtered list of tuples
            (wav_filename, wav_len, speaker)
        """
        min_len = self.hparams.input_length_range[0]
        max_len = self.hparams.input_length_range[1]
        filter_entries = []
        for items in self.entries:
            if self.kaldi_io_feats[items[0]].shape[0] in range(min_len, max_len):
                filter_entries.append(items)
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
            self.feature_normalizer.compute_cmvn_kaldiio(
                self.entries, self.speakers, self.kaldi_io_feats, feature_dim
            )
        self.feature_normalizer.save_cmvn(["speaker", "mean", "var"])
        return self
