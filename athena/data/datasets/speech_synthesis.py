# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Ruixiong Zhang
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
from athena.transform import AudioFeaturizer
from ..text_featurizer import TextFeaturizer
from ..feature_normalizer import FeatureNormalizer
from .base import BaseDatasetBuilder


class SpeechSynthesisDatasetBuilder(BaseDatasetBuilder):
    """ SpeechSynthesisDatasetBuilder

    Args:
        for __init__(self, config=None)

    Config:
        audio_config: the config file for feature extractor, default={'type':'Fbank'}
        vocab_file: the vocab file, default='data/utils/ch-en.vocab'

    Interfaces::
        __len__(self): return the number of data samples
        num_class(self): return the max_index of the vocabulary + 1
        @property:
          sample_shape:
            {"input": tf.TensorShape([None]),
             "input_length": tf.TensorShape([1]),
             "output_length": tf.TensorShape([1]),
             "output": tf.TensorShape([None, dimension]),
             "speaker": tf.TensorShape([1])}
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "text_config": {"type":"vocab", "model":"athena/utils/vocabs/ch-en.vocab"},
        "num_cmvn_workers": 1,
        "cmvn_file": None,
        "remove_unk": True,
        "input_length_range": [1, 10000],
        "output_length_range": [20, 50000],
        "data_csv": None,
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        self.speakers_dict = {}
        self.speakers_ids_dict = {}
        self.audio_featurizer = AudioFeaturizer(self.hparams.audio_config)
        self.feature_normalizer = FeatureNormalizer(self.hparams.cmvn_file)
        self.text_featurizer = TextFeaturizer(self.hparams.text_config)
        if self.hparams.data_csv is not None:
            self.preprocess_data(self.hparams.data_csv)

    def preprocess_data(self, file_path):
        """ Generate a list of tuples (wav_filename, wav_length_ms, transcript, speaker)."""
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

        speakers_ids = list(range(len(self.speakers)))
        self.speakers_dict = dict(zip(self.speakers, speakers_ids))
        self.speakers_ids_dict = dict(zip(speakers_ids, self.speakers))

        # handling special case for text_featurizer
        self.entries.sort(key=lambda item: len(item[2]))
        if self.text_featurizer.model_type == "text":
            _, _, all_transcripts, _ = zip(*self.entries)
            self.text_featurizer.load_model(all_transcripts)

        # apply some filter
        unk = self.text_featurizer.unk_index
        if self.hparams.remove_unk and unk != -1:
            self.entries = list(filter(lambda x: unk not in
                                self.text_featurizer.encode(x[2]), self.entries))
        self.entries = list(filter(lambda x: len(x[2]) in
                            range(self.hparams.input_length_range[0],
                            self.hparams.input_length_range[1]), self.entries))
        self.entries = list(filter(lambda x: float(x[1]) in
                            range(self.hparams.output_length_range[0],
                            self.hparams.output_length_range[1]), self.entries))
        return self

    def __getitem__(self, index):
        audio_data, _, transcripts, speaker = self.entries[index]
        audio_feat = self.audio_featurizer(audio_data)
        audio_feat = self.feature_normalizer(audio_feat, speaker)
        audio_feat_length = audio_feat.shape[0]
        audio_feat = tf.reshape(audio_feat, [audio_feat_length, -1])

        text = self.text_featurizer.encode(transcripts)
        text.append(self.text_featurizer.model.eos_index)
        text_length = len(text)
        return {
            "input": text,
            "input_length": text_length,
            "output_length": audio_feat_length,
            "output": audio_feat,
            "speaker": self.speakers_dict[speaker]
        }

    @property
    def num_class(self):
        """ return the max_index of the vocabulary + 1"""
        return len(self.text_featurizer)

    @property
    def feat_dim(self):
        """ return the number of feature dims """
        return self.audio_featurizer.dim

    @property
    def sample_type(self):
        return {
            "input": tf.int32,
            "input_length": tf.int32,
            "output_length": tf.int32,
            "output": tf.float32,
            "speaker": tf.int32
        }

    @property
    def sample_shape(self):
        feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        return {
            "input": tf.TensorShape([None]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None, feature_dim]),
            "speaker": tf.TensorShape([])
        }

    @property
    def sample_signature(self):
        feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        return (
            {
                "input": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None, feature_dim), dtype=tf.float32),
                "speaker": tf.TensorSpec(shape=(None), dtype=tf.int32)
            },
        )
