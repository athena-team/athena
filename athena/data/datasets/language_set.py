# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li
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
import tqdm
import tensorflow as tf
from ..text_featurizer import TextFeaturizer
from ...utils.hparam import register_and_parse_hparams
from .base import BaseDatasetBuilder

class LanguageDatasetBuilder(BaseDatasetBuilder):
    """ LanguageDatasetBuilder
    """
    default_config = {
        "input_text_config": None,
        "output_text_config": None,
        "input_length_range": [1, 1000],
        "output_length_range": [1, 1000]
    }
    def __init__(self, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.input_text_featurizer = TextFeaturizer(self.hparams.input_text_config)
        self.output_text_featurizer = TextFeaturizer(self.hparams.output_text_config)

    def load_csv(self, file_path):
        """ load csv file """
        logging.info("Loading data from {}".format(file_path))
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()
        lines = lines[1:]
        lines = [line.split("\t") for line in lines]
        entries = [tuple(line) for line in lines]
        input_transcripts, output_transcripts = zip(*entries)
        if self.input_text_featurizer.model_type == "text":
            self.input_text_featurizer.load_model(input_transcripts)
        if self.output_text_featurizer.model_type == "text":
            self.output_text_featurizer.load_model(output_transcripts)
        self.entries = []
        entries = tqdm.tqdm(entries)
        for input_transcript, output_transcript in entries:
            input_labels = self.input_text_featurizer.encode(input_transcript)
            output_labels = self.output_text_featurizer.encode(output_transcript)
            input_length = len(input_labels)
            output_length = len(output_labels)
            if input_length not in range(self.hparams.input_length_range[0],
                self.hparams.input_length_range[1]):
                continue
            if output_length not in range(self.hparams.output_length_range[0],
                self.hparams.output_length_range[1]):
                continue
            self.entries.append(tuple(
                [input_labels, input_length, output_labels, output_length]))

        self.entries.sort(key=lambda item: float(item[1]))

        return self

    def __getitem__(self, index):
        input_labels, input_length, output_labels, output_length = self.entries[index]

        return {
            "input": input_labels,
            "input_length": input_length,
            "output": output_labels,
            "output_length": output_length,
        }

    def __len__(self):
        """ return the number of data samples """
        return len(self.entries)

    @property
    def num_class(self):
        """ return the max_index of the vocabulary """
        return len(self.output_text_featurizer)

    @property
    def sample_type(self):
        return {
            "input": tf.int32,
            "input_length": tf.int32,
            "output": tf.int32,
            "output_length": tf.int32,
        }

    @property
    def sample_shape(self):
        return {
            "input": tf.TensorShape([None]),
            "input_length": tf.TensorShape([]),
            "output": tf.TensorShape([None]),
            "output_length": tf.TensorShape([]),
        }

    @property
    def sample_signature(self):
        return (
            {
                "input": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "input_length": tf.TensorSpec(shape=([None]), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=([None]), dtype=tf.int32),
            },
        )
