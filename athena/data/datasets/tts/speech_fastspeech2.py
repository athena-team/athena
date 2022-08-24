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
import numpy as np
from athena.transform import AudioFeaturizer

def average_by_duration(x, durs):
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))

    # calculate charactor f0/energy
    x_char = np.zeros((durs.shape[0]+1,), dtype=np.float32) #add eos
    for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values) if len(values) > 0 else 0.0  # np.mean([]) = nan.
    return x_char.astype(np.float32)

@tf.function(
    input_signature=[tf.TensorSpec(None, tf.float32), tf.TensorSpec(None, tf.int32)]
)
def tf_average_by_duration(x, durs):
    outs = tf.numpy_function(average_by_duration, [x, durs], tf.float32)
    return outs

class SpeechFastspeech2DatasetBuilder(BaseDatasetBuilder):
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
        self.audio_featurizer = AudioFeaturizer(self.hparams.audio_config)
        params_func = self.audio_featurizer.feat.params
        params = params_func(self.hparams.audio_config)
        self.frame_length = params.frame_length
        self.speakers_dict = {}
        self.speakers_ids_dict = {}
        self.feature_normalizer = FS2FeatureNormalizer(self.hparams.cmvn_file)
        self.text_featurizer = TextFeaturizer(self.hparams.text_config)
        if self.hparams.data_csv is not None:
            self.preprocess_data(self.hparams.data_csv)

    def load_duration(self, duration):
        duration = tf.convert_to_tensor([float(d) for d in duration])
        duration = duration / self.frame_length
        duration = tf.cast(tf.clip_by_value(tf.math.round(duration), 0.0,
                                            tf.cast(10000, dtype=tf.float32)), dtype=tf.int32)
        return duration

    def preprocess_data(self, file_path):
        """generate a list of tuples (audio_feature, wav_length_ms, transcript, duration, speaker).
        """
        logging.info("Loading data from {}".format(file_path))
        lines = pd.read_csv(file_path,"\t")
        headers = lines.keys()
        lines_num = lines.shape[0]
        self.entries = []
        self.speakers = []
        for l in range(lines_num):
            audio_feature = lines["audio_feature"][l]
            wav_len = lines["wav_length_ms"][l]
            transcript = lines["transcript"][l]
            duration = [] if "duration" not in headers else lines["duration"][l].split(' ')
            if len(duration) != 0:
                duration = self.load_duration(duration)
            speaker = "global" if "speaker" not in headers else lines["speaker"][l]
            self.entries.append(
                tuple([audio_feature, wav_len, transcript, duration, speaker])
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

    def load_audio_feature(self, audio_feature_file):
        audio_feature = np.load(audio_feature_file)  # audio_feature = {'mel':[array], 'f0':[array], 'energy':[array]}
        mel, f0, energy = audio_feature['mel'], audio_feature['f0'], audio_feature['energy']
        mel_tensor = tf.convert_to_tensor(mel, dtype=tf.float32)
        f0_tensor = tf.convert_to_tensor(f0, dtype=tf.float32)
        energy_tensor = tf.convert_to_tensor(energy, dtype=tf.float32)
        return mel_tensor, f0_tensor, energy_tensor

    def __getitem__(self, index):
        audio_feature_file, _, transcripts, durations, speaker = self.entries[index]
        utt_id = tf.convert_to_tensor(os.path.splitext(os.path.basename(audio_feature_file))[0])
        mel_tensor, f0_tensor, energy_tensor = self.load_audio_feature(audio_feature_file)
        mel_norm = self.feature_normalizer(mel_tensor, speaker, "mel")
        f0_norm = self.feature_normalizer(f0_tensor, speaker, "f0")
        energy_norm = self.feature_normalizer(energy_tensor, speaker, "energy")

        audio_feat_length = mel_norm.shape[0]
        mel_norm = tf.reshape(mel_norm, [audio_feat_length, -1])

        text = self.text_featurizer.encode(transcripts)
        text.append(self.text_featurizer.model.eos_index)
        text_length = len(text)

        duration_index = []
        for index, duration in enumerate(durations):
            duration_index.extend(list([index]) * int(duration))
        duration_index = duration_index[: audio_feat_length]
        if 0 < len(duration_index) < audio_feat_length:
            expanded_index = list([duration_index[-1]]) * int(audio_feat_length - len(duration_index))
            duration_index.extend(expanded_index)

        f0 = tf_average_by_duration(f0_norm, durations)
        energy = tf_average_by_duration(energy_norm, durations)

        return {
            "utt_id": utt_id,
            "input": text,
            "input_length": text_length,
            "output_length": audio_feat_length,
            "output": mel_norm,
            "speaker": self.speakers_dict[speaker],
            "f0": f0,
            "energy":  energy,
            "duration": duration_index
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
                "speaker": tf.int32,
                "duration": tf.int32
            }
        """
        return {
            "utt_id": tf.string,
            "input": tf.int32,
            "input_length": tf.int32,
            "output_length": tf.int32,
            "output": tf.float32,
            "f0": tf.float32,
            "energy": tf.float32, 
            "speaker": tf.int32,
            "duration": tf.int32
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
                "f0": tf.TensorShape([None]),
                "energy": tf.TensorShape([None]),
                "speaker": tf.TensorShape([]),
                "duration": tf.TensorShape([None])
            }
        """
        feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        return {
            "utt_id": tf.TensorShape([]),
            "input": tf.TensorShape([None]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None, feature_dim]),
            "f0": tf.TensorShape([None]),
            "energy": tf.TensorShape([None]),
            "speaker": tf.TensorShape([]),
            "duration": tf.TensorShape([None])
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
                "f0": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                "energy": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
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
                "f0": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                "energy": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                "speaker": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "duration": tf.TensorSpec(shape=(None, None), dtype=tf.int32)
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