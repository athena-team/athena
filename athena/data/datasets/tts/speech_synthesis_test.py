# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Cheng Wen
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
from athena.data.text_featurizer import TextFeaturizer
from athena.data.feature_normalizer import FS2FeatureNormalizer
from athena.data.datasets.base import BaseDatasetBuilder
import pandas as pd
import os

class SpeechSynthesisTestDatasetBuilder(BaseDatasetBuilder):
    """SpeechSynthesisDatasetBuilder
    """
    default_config = {
        "text_config": {"type":"eng_vocab", "model":"examples/tts/data_baker/data/vocab"},
        "num_cmvn_workers": 1,
        "cmvn_file": None,
        "remove_unk": True,
        "input_length_range": [1, 10000],
        "output_length_range": [20, 50000],
        "data_csv": None,
    }

    def __init__(self, data_csv, config=None):
        super().__init__(config=config)
        self.speakers_dict = {}
        self.speakers_ids_dict = {}
        self.feature_normalizer = FS2FeatureNormalizer(self.hparams.cmvn_file)
        self.text_featurizer = TextFeaturizer(self.hparams.text_config)
        self.preprocess_data(data_csv)

    def preprocess_data(self, file_path):
        """generate a list of tuples (utt_id, transcript).
        """
        logging.info("Loading data from {}".format(file_path))
        lines = pd.read_csv(file_path,"\t",dtype={'utt_id':str})
        headers = lines.keys()
        lines_num = lines.shape[0]
        self.entries = []
        self.speakers = []
        for l in range(lines_num):
            utt_id = str(lines["utt_id"][l])
            transcript = lines["transcript"][l]
            speaker = "global" if "speaker" not in headers else lines["speaker"][l]
            self.entries.append(
                tuple([utt_id, transcript, speaker])
            )
            if speaker not in self.speakers:
                self.speakers.append(speaker)

        speakers_ids = list(range(len(self.speakers)))
        self.speakers_dict = dict(zip(self.speakers, speakers_ids))
        self.speakers_ids_dict = dict(zip(speakers_ids, self.speakers))

        # handling special case for text_featurizer
        self.entries.sort(key=lambda item: len(item[2]))
        if self.text_featurizer.model_type == "text":
            _, _, all_transcripts, _, _ = zip(*self.entries)
            self.text_featurizer.load_model(all_transcripts)

        # apply some filter
        unk = self.text_featurizer.unk_index
        if self.hparams.remove_unk and unk != -1:
            self.entries = list(filter(lambda x: unk not in
                                self.text_featurizer.encode(x[1]), self.entries))
        self.entries = list(filter(lambda x: len(x[1]) in
                            range(self.hparams.input_length_range[0],
                            self.hparams.input_length_range[1]), self.entries))
        return self

    def __getitem__(self, index):
        utt_id, transcripts, speaker = self.entries[index]
        utt_id = tf.convert_to_tensor(utt_id)

        text = self.text_featurizer.encode(transcripts)
        text.append(self.text_featurizer.model.eos_index)
        text_length = len(text)

        return {
            "utt_id": utt_id,
            "input": text,
            "input_length": text_length,
            "speaker": self.speakers_dict[speaker],
        }

    @property
    def num_class(self):
        """:obj:`@property`

        Returns:
            int: the max_index of the vocabulary
        """
        return len(self.text_featurizer)

    # @property
    # def feat_dim(self):
    #     """return the number of feature dims
    #     """
    #     return self.audio_featurizer.dim

    @property
    def sample_type(self):
        """:obj:`@property`

        Returns:
            dict: sample_type of the dataset::

            {
                "utt_id": tf.string,
                "input": tf.int32,
                "input_length": tf.int32,
                "speaker": tf.int32,             
            }
        """
        return {
            "utt_id": tf.string,
            "input": tf.int32,
            "input_length": tf.int32,          
            "speaker": tf.int32,
        }

    @property
    def sample_shape(self):
        """:obj:`@property`

        Returns:
            dict: sample_shape of the dataset::

            {
                "utt_id": tf.TensorShape([]),
                "input": tf.TensorShape([None]),
                "input_length": tf.TensorShape([]),              
                "speaker": tf.TensorShape([]),        
            }
        """
        return {
            "utt_id": tf.TensorShape([]),
            "input": tf.TensorShape([None]),
            "input_length": tf.TensorShape([]),         
            "speaker": tf.TensorShape([]),
        }

    @property
    def sample_signature(self):
        """:obj:`@property`

        Returns:
            dict: sample_signature of the dataset::

            {
                "utt_id": tf.TensorSpec(shape=(None), dtype=tf.string),
                "input": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),              
                "speaker": tf.TensorSpec(shape=(None), dtype=tf.int32)
            }
        """
        return (
            {
                "utt_id": tf.TensorSpec(shape=(None), dtype=tf.string),
                "input": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),             
                "speaker": tf.TensorSpec(shape=(None), dtype=tf.int32),
            },
        )

    def compute_cmvn_if_necessary(self, is_necessary=True):
            """compute cmvn file
            """
            if not is_necessary:
                return self
            if os.path.exists(self.hparams.cmvn_file):
                return self

            with tf.device("/cpu:0"):
                self.feature_normalizer.compute_fs2_cmvn(
                    self.entries, self.speakers, self.hparams.num_cmvn_workers) 
            self.feature_normalizer.save_cmvn(
                ["audio_feature", "mean", "var"])
            return self