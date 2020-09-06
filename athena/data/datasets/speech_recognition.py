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
from ..text_featurizer import TextFeaturizer
from .base import SpeechBaseDatasetBuilder


class SpeechRecognitionDatasetBuilder(SpeechBaseDatasetBuilder):
    """SpeechRecognitionDatasetBuilder
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "text_config": {"type":"vocab", "model":"athena/utils/vocabs/ch-en.vocab"},
        "num_cmvn_workers": 1,
        "cmvn_file": None,
        "remove_unk": True,
        "input_length_range": [20, 50000],
        "output_length_range": [1, 10000],
        "speed_permutation": [1.0],
        "data_csv": None,
        "words": None
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        self.text_featurizer = TextFeaturizer(self.hparams.text_config)
        if self.hparams.data_csv is not None:
            self.preprocess_data(self.hparams.data_csv)

    def preprocess_data(self, file_path):
        """generate a list of tuples (wav_filename, wav_length_ms, transcript, speaker).
        """
        logging.info("Loading data from {}".format(file_path))
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()
        headers = lines[0]
        lines = lines[1:]
        lines = [line.split("\t") for line in lines]
        self.entries = [tuple(line) for line in lines]

        # handling speakers
        self.speakers = []
        if "speaker" not in headers.split("\t"):
            entries = self.entries
            self.entries = []
            for wav_filename, wav_len, transcripts in entries:
                self.entries.append(
                    tuple([wav_filename, wav_len, transcripts, "global"])
                )
            self.speakers.append("global")
        else:
            for _, _, _, speaker in self.entries:
                if speaker not in self.speakers:
                    self.speakers.append(speaker)

        # handling speed
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

        # handling special case for text_featurizer
        self.entries.sort(key=lambda item: float(item[1]))
        if self.text_featurizer.model_type == "text":
            _, _, all_transcripts, _, _ = zip(*self.entries)
            self.text_featurizer.load_model(all_transcripts)

        # apply some filter
        unk = self.text_featurizer.unk_index
        if self.hparams.remove_unk and unk != -1:
            self.entries = list(filter(lambda x: unk not in
                                self.text_featurizer.encode(x[2]), self.entries))
        self.entries = list(filter(lambda x: int(x[1]) in
                            range(self.hparams.input_length_range[0],
                            self.hparams.input_length_range[1]), self.entries))
        self.entries = list(filter(lambda x: len(x[2]) in
                            range(self.hparams.output_length_range[0],
                            self.hparams.output_length_range[1]), self.entries))
        return self

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
            }
        """
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

