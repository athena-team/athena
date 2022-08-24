# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Ruixiong Zhang; Cheng Wen
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
from athena.data.datasets.base import SpeechBaseDatasetBuilder
import pandas as pd
import os

class SpeechSynthesisDatasetBuilder(SpeechBaseDatasetBuilder):
    """SpeechSynthesisDatasetBuilder
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
        assert self.audio_featurizer.feat is not None
        assert self.hparams.audio_config is not None
        params_func = self.audio_featurizer.feat.params
        params = params_func(self.hparams.audio_config)
        self.frame_length = params.frame_length
        self.speakers_dict = {}
        self.speakers_ids_dict = {}
        self.text_featurizer = TextFeaturizer(self.hparams.text_config)
        if self.hparams.data_csv is not None:
            self.preprocess_data(self.hparams.data_csv)

    def preprocess_data(self, file_path):
        """generate a list of tuples (wav_filename, wav_length_ms, transcript, speaker).
        """
        logging.info("Loading data from {}".format(file_path))
        lines = pd.read_csv(file_path,"\t")
        headers = lines.keys()
        lines_num = lines.shape[0]
        self.entries = []
        self.speakers = []
        for l in range(lines_num):
            wav_filename = lines["wav_filename"][l]
            wav_length = lines["wav_length_ms"][l]
            transcript = lines["transcript"][l]
            speaker = "global" if "speaker" not in headers else lines["speaker"][l]
            self.entries.append(
                tuple([wav_filename, wav_length, transcript, speaker])
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
        utt_id = tf.convert_to_tensor(os.path.splitext(os.path.basename(audio_data))[0])
        audio_feat = self.audio_featurizer(audio_data)
        audio_feat = self.feature_normalizer(audio_feat, speaker)
        audio_feat_length = audio_feat.shape[0]
        audio_feat = tf.reshape(audio_feat, [audio_feat_length, -1])

        text = self.text_featurizer.encode(transcripts)
        text.append(self.text_featurizer.model.eos_index)
        text_length = len(text)
        return {
            "utt_id": utt_id,
            "input": text,
            "input_length": text_length,
            "output_length": audio_feat_length,
            "output": audio_feat,
            "speaker": self.speakers_dict[speaker],
        }

    @property
    def num_class(self):
        """:obj:`@property`

        Returns:
            int: the max_index of the vocabulary
        """
        return len(self.text_featurizer)

    @property
    def feat_dim(self):
        """return the number of feature dims
        """
        return self.audio_featurizer.dim

    @property
    def sample_type(self):
        """:obj:`@property`

        Returns:
            dict: sample_type of the dataset::

            {
                "utt_id": tf.string,
                "input": tf.int32,
                "input_length": tf.int32,
                "output_length": tf.int32,
                "output": tf.float32,
                "speaker": tf.int32
            }
        """
        return {
            "utt_id": tf.string,
            "input": tf.int32,
            "input_length": tf.int32,
            "output_length": tf.int32,
            "output": tf.float32,
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
                "output_length": tf.TensorShape([]),
                "output": tf.TensorShape([None, feature_dim]),
                "speaker": tf.TensorShape([])
            }
        """
        feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        return {
            "utt_id": tf.TensorShape([]),
            "input": tf.TensorShape([None]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None, feature_dim]),
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
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None, feature_dim),
                                        dtype=tf.float32),
                "speaker": tf.TensorSpec(shape=(None), dtype=tf.int32)
            }
        """
        feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        return (
            {
                "utt_id": tf.TensorSpec(shape=(None), dtype=tf.string),
                "input": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None, feature_dim), dtype=tf.float32),
                "speaker": tf.TensorSpec(shape=(None), dtype=tf.int32),
            },
        )

