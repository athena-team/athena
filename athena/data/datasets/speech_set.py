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

from absl import logging
import tensorflow as tf
from .base import SpeechBaseDatasetBuilder


class SpeechDatasetBuilder(SpeechBaseDatasetBuilder):
    """SpeechDatasetBuilder
    """

    default_config = {
        "audio_config": {"type": "Fbank"},
        "num_cmvn_workers": 1,
        "cmvn_file": None,
        "input_length_range": [20, 50000],
        "data_csv": None,
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        if self.hparams.data_csv is not None:
            self.preprocess_data(self.hparams.data_csv)

    def preprocess_data(self, file_path):
        """generate a list of tuples (wav_filename, wav_length_ms, speaker)."""
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

        self.entries = list(filter(lambda x: int(x[1]) in
                            range(self.hparams.input_length_range[0],
                            self.hparams.input_length_range[1]), self.entries))
        return self

    def __getitem__(self, index):
        """get a sample

        Args:
            index (int): index of the entries

        Returns:
            dict: sample::

            {
                "input": input_data,
                "input_length": input_data.shape[0],
                "output": output_data,
                "output_length": output_data.shape[0],
            }
        """
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

    @property
    def num_class(self):
        """:obj:`@property`

        Returns:
            int: the target dim
        """
        target_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        return target_dim

    @property
    def sample_type(self):
        """:obj:`@property`

        Returns:
            dict: sample_type of the dataset::

            {
                "input": tf.float32,
                "input_length": tf.int32,
                "output": tf.float32,
                "output_length": tf.int32,
            }
        """
        return {
            "input": tf.float32,
            "input_length": tf.int32,
            "output": tf.float32,
            "output_length": tf.int32,
        }

    @property
    def sample_shape(self):
        """:obj:`@property`

        Returns:
            dict: sample_shape of the dataset::

            {
                "input": tf.TensorShape(
                    [None, self.audio_featurizer.dim, self.audio_featurizer.num_channels]
                ),
                "input_length": tf.TensorShape([]),
                "output": tf.TensorShape([None, None]),
                "output_length": tf.TensorShape([]),
            }
        """
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
        """:obj:`@property`

        Returns:
            dict: sample_signature of the dataset::

            {
                "input": tf.TensorSpec(
                    shape=(None, None, None, None), dtype=tf.float32
                ),
                "input_length": tf.TensorSpec(shape=([None]), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                "output_length": tf.TensorSpec(shape=([None]), dtype=tf.int32),
            }
        """
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
