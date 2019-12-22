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
from ..text_featurizer import TextFeaturizer
from ..feature_normalizer import FeatureNormalizer
from .base import BaseDatasetBuilder


class SpeechRecognitionDatasetBuilder(BaseDatasetBuilder):
    """ SpeechRecognitionDatasetBuilder

    Args:
        for __init__(self, config=None)

    Config::
        audio_config: the config file for feature extractor, default={'type':'Fbank'}
        vocab_file: the vocab file, default='data/utils/ch-en.vocab'

    Interfaces::
        __len__(self): return the number of data samples
        num_class(self): return the max_index of the vocabulary + 1
        @property:
          sample_shape:
            {"input": tf.TensorShape([None, self.audio_featurizer.dim,
                                  self.audio_featurizer.num_channels]),
             "input_length": tf.TensorShape([1]),
             "output_length": tf.TensorShape([1]),
             "output": tf.TensorShape([None])}
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "text_config": {"type":"vocab", "model":"athena/utils/vocabs/ch-en.vocab"},
        "cmvn_file": None,
        "remove_unk": True,
        "input_length_range": [20, 50000],
        "output_length_range": [1, 10000],
        "speed_permutation": [1.0],
    }

    def __init__(self, config=None):
        super().__init__()
        # hparams
        self.hparams = register_and_parse_hparams(
            self.default_config, config, cls=self.__class__)
        logging.info("hparams: {}".format(self.hparams))

        self.audio_featurizer = AudioFeaturizer(self.hparams.audio_config)
        self.feature_normalizer = FeatureNormalizer(self.hparams.cmvn_file)
        self.text_featurizer = TextFeaturizer(self.hparams.text_config)

    def reload_config(self, config):
        """ reload the config """
        if config is not None:
            self.hparams.override_from_dict(config)

    def preprocess_data(self, file_path):
        """ Generate a list of tuples (wav_filename, wav_length_ms, transcript speaker)."""
        logging.info("Loading data from {}".format(file_path))
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()
        headers = lines[0]
        lines = lines[1:]
        lines = [line.split("\t") for line in lines]
        self.entries = [tuple(line) for line in lines]

        self.speakers = []
        if "speaker" not in headers.split("\t"):
            entries = self.entries
            self.entries = []
            if self.text_featurizer.model_type == "text":
                _, _, all_transcripts = zip(*entries)
                self.text_featurizer.load_model(all_transcripts)
            for wav_filename, wav_len, transcripts in entries:
                self.entries.append(
                    tuple([wav_filename, wav_len, transcripts, "global"])
                )
            self.speakers.append("global")
        else:
            if self.text_featurizer.model_type == "text":
                _, _, all_transcripts, _ = zip(*entries)
                self.text_featurizer.load_model(all_transcripts)
            for _, _, _, speaker in self.entries:
                if speaker not in self.speakers:
                    self.speakers.append(speaker)

        entries = self.entries
        self.entries = []
        if len(self.hparams.speed_permutation) > 1:
            logging.info("perform speed permutation")
        for speed in self.hparams.speed_permutation:
            for wav_filename, wav_len, transcripts, speaker in entries:
                self.entries.append(
                    tuple([wav_filename,
                    float(wav_len) / float(speed), transcripts, speed, speaker
                ]))

        self.entries.sort(key=lambda item: float(item[1]))

        # apply some filter
        self.filter_sample_by_unk()
        self.filter_sample_by_input_length()
        self.filter_sample_by_output_length()
        return self

    def load_csv(self, file_path):
        """ load csv file """
        return self.preprocess_data(file_path)

    def __getitem__(self, index):
        audio_data, _, transcripts, speed, speaker = self.entries[index]
        feat = self.audio_featurizer(audio_data, speed=speed)
        feat = self.feature_normalizer(feat, speaker)
        feat_length = feat.shape[0]

        label = self.text_featurizer.encode(transcripts)
        label_length = len(label)
        return {
            "input": feat,
            "input_length": feat_length,
            "output_length": label_length,
            "output": label,
        }

    def __len__(self):
        """ return the number of data samples """
        return len(self.entries)

    @property
    def num_class(self):
        """ return the max_index of the vocabulary + 1"""
        return len(self.text_featurizer)

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
            "output_length": tf.int32,
            "output": tf.int32,
        }

    @property
    def sample_shape(self):
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return {
            "input": tf.TensorShape([None, dim, nc]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None]),
        }

    @property
    def sample_signature(self):
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

    def filter_sample_by_unk(self):
        """filter samples which contain unk
        """
        if self.hparams.remove_unk is False:
            return self
        filter_entries = []
        unk = self.text_featurizer.unk_index
        if unk == -1:
            return self
        for items in self.entries:
            if unk not in self.text_featurizer.encode(items[2]):
                filter_entries.append(items)
        self.entries = filter_entries
        return self

    def filter_sample_by_input_length(self):
        """filter samples by input length

        The length of filterd samples will be in [min_length, max_length)

        Args:
            self.hparams.input_length_range = [min_len, max_len]
            min_len: the minimal length(ms)
            max_len: the maximal length(ms)
        returns:
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

    def filter_sample_by_output_length(self):
        """filter samples by output length

        The length of filterd samples will be in [min_length, max_length)

        Args:
            self.hparams.output_length_range = [min_len, max_len]
            min_len: the minimal length
            max_len: the maximal length
        returns:
            entries: a filtered list of tuples
            (wav_filename, wav_len, transcripts, speed, speaker)
        """
        min_len = self.hparams.output_length_range[0]
        max_len = self.hparams.output_length_range[1]
        filter_entries = []
        for items in self.entries:
            if len(items[2]) in range(min_len, max_len):
                filter_entries.append(items)
        self.entries = filter_entries
        return self

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
