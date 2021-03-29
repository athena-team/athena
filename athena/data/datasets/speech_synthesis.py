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
from .base import SpeechBaseDatasetBuilder
import numpy as np


def average_by_duration(x, durs):
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))

    # calculate charactor f0/energy
    x_char = np.zeros((durs.shape[0],), dtype=np.float32)
    for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values) if len(values) > 0 else 0.0  # np.mean([]) = nan.
    return x_char.astype(np.float32)

def tf_average_by_duration(x, durs):
    outs = tf.numpy_function(average_by_duration, [x, durs], tf.float32)
    return outs

def _norm_mean_std(x, mean, std):
    zero_idxs = np.where(x == 0.0)[0]
    x = (x - mean) / std
    x[zero_idxs] = 0.0
    return x

def _norm_mean_std_tf(x, mean, std):
    outs = _norm_mean_std(x, mean, std)
    return outs

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

        energy_config = {
            "type": "Framepow",
            "window_length": params.window_length,
            "frame_length": params.frame_length,
            "remove_dc_offset": params.remove_dc_offset
        }
        self.energy_featurizer = AudioFeaturizer(energy_config)

        pitch_config = {
            "type": "Pitch",
            "window_length": params.window_length,
            "frame_length": params.frame_length,
            "preEph_coeff": params.preEph_coeff,
            "upper_frequency_limit": params.upper_frequency_limit
        }
        self.pitch_featurizer = AudioFeaturizer(pitch_config)

        self.frame_length = params.frame_length
        self.speakers_dict = {}
        self.speakers_ids_dict = {}
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
        """generate a list of tuples (wav_filename, wav_length_ms, transcript, speaker).
        """
        logging.info("Loading data from {}".format(file_path))
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()
        headers = lines[0]
        lines = lines[1:]
        self.entries = []
        self.speakers = []
        for line in lines:
            entry = line.split("\t")
            duration = [] if "duration" not in headers.split("\t") else entry[3].split(' ')
            if len(duration) != 0:
                duration = self.load_duration(duration)
            speaker = "global" if "speaker" not in headers.split("\t") else entry[-1]
            self.entries.append(
                tuple([entry[0], entry[1], entry[2], duration, speaker])
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
        audio_data, _, transcripts, durations, speaker = self.entries[index]
        audio_feat = self.audio_featurizer(audio_data)
        audio_feat = self.feature_normalizer(audio_feat, speaker)
        audio_feat_length = audio_feat.shape[0]
        audio_feat = tf.reshape(audio_feat, [audio_feat_length, -1])

        pitch = self.pitch_featurizer(audio_data)[:, 0]

        energy = self.energy_featurizer(audio_data)
        #energy = energy / 32768

        pitch_len = tf.shape(pitch)[0]
        audio_feat_length = tf.shape(audio_feat)[0]
        pitch = pitch[:audio_feat_length]
        energy = energy[:audio_feat_length]
        audio_feat = audio_feat[:pitch_len]

        text = self.text_featurizer.encode(transcripts)
        text_length = len(text)

        duration_index = []
        for index, duration in enumerate(durations):
            duration_index.extend(list([index]) * int(duration))
        duration_index = duration_index[: audio_feat_length]
        if 0 < len(duration_index) < audio_feat_length:
            expanded_index = list([duration_index[-1]]) * int(audio_feat_length - len(duration_index))
            duration_index.extend(expanded_index)
        pitch = tf_average_by_duration(tf.convert_to_tensor(pitch), tf.convert_to_tensor(durations))
        energy = tf_average_by_duration(tf.convert_to_tensor(energy), tf.convert_to_tensor(durations))
        return {
            "input": text,
            "input_length": text_length,
            "output_length": audio_feat_length,
            "output": audio_feat,
            "speaker": self.speakers_dict[speaker],
            "pitch": pitch,
            "energy": energy,
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
                "input": tf.int32,
                "input_length": tf.int32,
                "output_length": tf.int32,
                "output": tf.float32,
                "speaker": tf.int32
            }
        """
        return {
            "input": tf.int32,
            "input_length": tf.int32,
            "output_length": tf.int32,
            "output": tf.float32,
            "speaker": tf.int32,
            "pitch": tf.float32,
            "energy": tf.float32,
            "duration": tf.int32
        }

    @property
    def sample_shape(self):
        """:obj:`@property`

        Returns:
            dict: sample_shape of the dataset::

            {
                "input": tf.TensorShape([None]),
                "input_length": tf.TensorShape([]),
                "output_length": tf.TensorShape([]),
                "output": tf.TensorShape([None, feature_dim]),
                "speaker": tf.TensorShape([])
            }
        """
        feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        return {
            "input": tf.TensorShape([None]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None, feature_dim]),
            "speaker": tf.TensorShape([]),
            "pitch": tf.TensorShape([None]),
            "energy": tf.TensorShape([None]),
            "duration": tf.TensorShape([None])
        }

    @property
    def sample_signature(self):
        """:obj:`@property`

        Returns:
            dict: sample_signature of the dataset::

            {
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
                "input": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None, feature_dim), dtype=tf.float32),
                "speaker": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "pitch": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                "energy": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                "duration": tf.TensorSpec(shape=(None, None), dtype=tf.int32)
            },
        )
