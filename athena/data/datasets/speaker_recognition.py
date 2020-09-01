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

from absl import logging
import tensorflow as tf
from .base import SpeechBaseDatasetBuilder

class SpeakerRecognitionDatasetBuilder(SpeechBaseDatasetBuilder):
    """SpeakerRecognitionDatasetBuilder
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "num_cmvn_workers": 1,
        "cmvn_file": None,
        "cut_frame": None,
        "input_length_range": [20, 50000],
        "data_csv": None
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        if self.hparams.data_csv is not None:
            self.preprocess_data(self.hparams.data_csv)

    def preprocess_data(self, file_path):
        """ Generate a list of tuples
            (wav_filename, wav_length_ms, speaker_id, speaker_name).
        """
        logging.info("Loading data csv {}".format(file_path))
        with open(file_path, "r", encoding='utf-8') as data_csv:
            lines = data_csv.read().splitlines()[1:]
        lines = [line.split("\t", 3) for line in lines]
        lines.sort(key=lambda item: int(item[1]))
        self.entries = [tuple(line) for line in lines]
        self.speakers = list(set(line[-1] for line in lines))

        # apply input length filter
        self.entries = list(filter(lambda x: int(x[1]) in
                            range(self.hparams.input_length_range[0],
                            self.hparams.input_length_range[1]), self.entries))
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

    @property
    def num_class(self):
        ''' return the number of speakers '''
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


class SpeakerVerificationDatasetBuilder(SpeakerRecognitionDatasetBuilder):
    """SpeakerVerificationDatasetBuilder
    """

    def __init__(self, config=None):
        super().__init__(config=config)

    def preprocess_data(self, file_path):
        """ Generate a list of tuples
            (wav_filename_a, speaker_a, wav_filename_b, speaker_b, label).
        """
        logging.info("Loading data csv {}".format(file_path))
        with open(file_path, "r", encoding='utf-8') as data_csv:
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
