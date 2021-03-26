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
    #print(x, mean, std)
    outs = _norm_mean_std(x, mean, std)
    #outs = tf.numpy_function(_norm_mean_std, [x, mean, std], tf.float32)
    return outs

class SpeechSynthesisDatasetBuilder(SpeechBaseDatasetBuilder):
    """SpeechSynthesisDatasetBuilder
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "text_config": {"type": "vocab","model": "athena/utils/vocabs/ch-en.vocab"},
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

    def load_duration(self, duration, maxtime):
        duration = [float(d) / self.frame_length for d in duration]
        total_frame = round(int(maxtime) / 1000 / self.frame_length)
        duration[-1] = total_frame - sum(duration[:-1])
        duration = tf.convert_to_tensor(duration)
        duration = tf.cast(tf.clip_by_value(tf.math.round(duration), 0.0,
                                            tf.cast(10000, dtype=tf.float32)),dtype=tf.int32)

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
            header_list = headers.split("\t")
            duration = [] if "duration" not in header_list \
                else entry[header_list.index("duration")].split(" ")
            if len(duration) != 0:
                duration = self.load_duration(duration, entry[1])
            f0_file = None if "f0_file" not in header_list \
                else entry[header_list.index("f0_file")]
            energy_file = None if "energy_file" not in header_list \
                else entry[header_list.index("energy_file")]
            speaker = "global" if "speaker" not in header_list \
                else entry[header_list.index("speaker")]
            self.entries.append(
                tuple([
                    entry[0], entry[1], entry[2], duration, f0_file, energy_file,
                    speaker
                ]))
            if speaker not in self.speakers:
                self.speakers.append(speaker)

        speakers_ids = list(range(len(self.speakers)))
        self.speakers_dict = dict(zip(self.speakers, speakers_ids))
        self.speakers_ids_dict = dict(zip(speakers_ids, self.speakers))

        # handling special case for text_featurizer
        self.entries.sort(key=lambda item: len(item[2].split(' ')))
        if self.text_featurizer.model_type == "text":
            _, _, all_transcripts, _, _, _ = zip(*self.entries)
            self.text_featurizer.load_model(all_transcripts)

        # apply some filter
        unk = self.text_featurizer.unk_index
        if self.hparams.remove_unk and unk != -1:
            self.entries = list(
                filter(lambda x: unk not in self.text_featurizer.encode(x[2]),
                       self.entries))
        self.entries = list(
            filter(
                lambda x: len(x[2].split(' ')) in range(
                    self.hparams.input_length_range[0], self.hparams.
                    input_length_range[1]), self.entries))
        self.entries = list(
            filter(
                lambda x: float(x[1]) in range(
                    self.hparams.output_length_range[0], self.hparams.
                    output_length_range[1]), self.entries))
        return self

    def __getitem__(self, index):
        audio_data, maxtime, transcripts, durations, f0_file, energy_file, speaker = self.entries[index]
        audio_feat = self.audio_featurizer(audio_data)
        if '.ark' not in audio_data and '.npy' not in audio_data:
            audio_feat = self.feature_normalizer(audio_feat, speaker)
        energy = tf.zeros([1])
        f0 = tf.zeros([1])
        if f0_file is not None and energy_file is not None:
            audio_feat_length = audio_feat.shape[0]
            audio_feat = tf.reshape(audio_feat, [audio_feat_length, -1])
            f0 = np.array(np.load(f0_file))
            energy = np.array(np.load(energy_file))

            f0_len = tf.shape(f0)[0]
            audio_feat_length = tf.shape(audio_feat)[0]
            f0 = f0[:audio_feat_length]
            energy = energy[:audio_feat_length]
            audio_feat = audio_feat[:f0_len]
        audio_feat_length = tf.shape(audio_feat)[0]
        text = self.text_featurizer.encode(transcripts)
        #text.append(self.text_featurizer.model.eos_index)
        text_length = len(text)

        duration_index = []
        for index, duration in enumerate(durations):
            duration_index.extend(list([index]) * int(duration))
        duration_index = duration_index[:audio_feat_length]
        if 0 < len(duration_index) < audio_feat_length:
            expanded_index = list([duration_index[-1]]) * int(audio_feat_length - len(duration_index))
            duration_index.extend(expanded_index)
        #ugly code: 243.02415466308594 and 50.29555130004883 is the f0's mean and std 
        f0 = tf.numpy_function(_norm_mean_std_tf, [tf.convert_to_tensor(f0), tf.convert_to_tensor(243.02415466308594), tf.convert_to_tensor(50.29555130004883)], tf.float32)
        f0 = tf_average_by_duration(tf.convert_to_tensor(f0), tf.convert_to_tensor(durations))
        #ugly code: same
        energy = tf.numpy_function(_norm_mean_std_tf, [tf.convert_to_tensor(energy), tf.convert_to_tensor(20.519119262695312), tf.convert_to_tensor(19.576580047607422)], tf.float32)
        energy = tf_average_by_duration(tf.convert_to_tensor(energy), tf.convert_to_tensor(durations))

        return {
            "input": text,
            "input_length": text_length,
            "output_length": audio_feat_length,
            "output": audio_feat,
            "speaker": self.speakers_dict[speaker],
            "pitch": f0,
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

    @property
    def mpc_sample_signature(self):
        return (
            {
                "input": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None, self.num_class), dtype=tf.float32),
                "speaker": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "pitch": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                "energy": tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                "duration": tf.TensorSpec(shape=(None, None), dtype=tf.int32)
            },
        )
