# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Chunxin Xiao; Dongwei Jiang
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
""" voice conversion dataset """

import os
import random
import librosa
import numpy as np
from absl import logging
import tensorflow as tf
import pyworld
from ..feature_normalizer import WorldFeatureNormalizer
from .base import BaseDatasetBuilder


class VoiceConversionDatasetBuilder(BaseDatasetBuilder):
    """VoiceConversionDatasetBuilder
    """
    default_config = {
        "cmvn_file": None,
        "input_length_range": [20, 50000],
        "data_csv": None,
        "num_cmvn_workers": 1,
        "enable_load_from_disk": True,
        "codedsp_dim": 36,
        "fs": 16000,
        "fft_size": 1024,
        "num_speakers": 0
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        self.sp_dim = self.hparams.codedsp_dim
        self.fs = self.hparams.fs
        self.fft_size = self.hparams.fft_size
        self.enable_load_from_disk = self.hparams.enable_load_from_disk
        self.feature_normalizer = WorldFeatureNormalizer(self.hparams.cmvn_file)
        self.entries_person_wavs = {}
        self.speakers = []
        if self.hparams.data_csv is not None:
            self.preprocess_data(self.hparams.data_csv)

        # to deal with the problem that train/eval have different number of speakers
        if self.hparams.num_speakers != 0:
            self.spk_num = self.hparams.num_speakers
        else:
            self.spk_num = len(self.speakers)
        # create one-hot speaker embedding
        self.spk_one_hot = {}
        onehot_arr = tf.eye(self.spk_num, dtype=tf.float32)
        spk_count = 0
        for name in self.speakers:
            self.spk_one_hot[name] = tf.gather(onehot_arr, axis=0, indices=[spk_count])
            spk_count = spk_count + 1

        self.speakers_dict = {}
        self.speakers_ids_dict = {}
        speakers_ids = list(range(len(self.speakers)))
        self.speakers_dict = dict(zip(self.speakers, speakers_ids))
        self.speakers_ids_dict = dict(zip(speakers_ids, self.speakers))

    def preprocess_data(self, file_path):
        """generate a list of tuples
           (src_wav_filename, src_speaker, tar_wav_filename, tar_speaker).
        """
        logging.info("Loading data from {}".format(file_path))
        with open(file_path, "r", encoding='utf-8') as file:
            lines = file.read().splitlines()
        lines = lines[1:]
        # In the VC task, it is necessary to construct the paired training corpus of
        # the original speaker and the target speaker and the multi-segment speech of
        # the speaker entries
        for line in lines:
            items = line.split("\t")
            src_wav_filename, src_speaker = items[0], items[1]
            if src_speaker not in self.speakers:
                self.entries_person_wavs[src_speaker] = []
                self.speakers.append(src_speaker)
            self.entries_person_wavs[src_speaker].append(src_wav_filename)
        for line in lines:
            items = line.split("\t")
            src_wav_filename, src_speaker = items[0], items[1]
            self.speakers.remove(src_speaker)
            tar_speaker = random.choice(self.speakers)
            self.speakers.append(src_speaker)
            tar_wav_filename = random.choice(self.entries_person_wavs[tar_speaker])
            self.entries.append(tuple([src_wav_filename, src_speaker,
                                       tar_wav_filename, tar_speaker]))
        return self

    def world_feature_extract(self, wav):
        """extract world feature

        World Vocoder parameterizes speech into three components::

            Pitch (fundamental frequency, F0) contour
            Harmonic spectral envelope(sp)
            Aperiodic spectral envelope (relative to the harmonic spectral envelope, ap)
        Refer to the address：https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
        """
        f0, timeaxis = pyworld.harvest(wav, self.fs)
        sp = pyworld.cheaptrick(wav, f0, timeaxis, self.fs, fft_size=self.fft_size)
        ap = pyworld.d4c(wav, f0, timeaxis, self.fs, fft_size=self.fft_size)
        coded_sp = pyworld.code_spectral_envelope(sp, self.fs, self.sp_dim)

        f0, sp, ap, coded_sp = tf.convert_to_tensor(f0, dtype=tf.float32), \
            tf.convert_to_tensor(sp, dtype=tf.float32), \
            tf.convert_to_tensor(ap, dtype=tf.float32), \
            tf.convert_to_tensor(coded_sp, dtype=tf.float32)
        return f0, sp, ap, coded_sp

    def load_from_disk(self, src_wav_filename, tar_wav_filename):
        """Calculting world features on-the-fly takes too much time,
           this is another option to load vc-related features from disk
        """
        src = np.load(src_wav_filename)
        tar = np.load(tar_wav_filename)
        src_f0, src_coded_sp, src_ap = src["f0"], src["coded_sp"], src["ap"]
        tar_f0, tar_coded_sp, tar_ap = tar["f0"], tar["coded_sp"], tar["ap"]
        tar_coded_sp = tar_coded_sp.T
        src_coded_sp = src_coded_sp.T
        src_f0, src_coded_sp, src_ap, tar_coded_sp = tf.convert_to_tensor(src_f0, dtype=tf.float32), \
            tf.convert_to_tensor(src_coded_sp, dtype=tf.float32), \
            tf.convert_to_tensor(src_ap, dtype=tf.float32), \
            tf.convert_to_tensor(tar_coded_sp, dtype=tf.float32)
        return src_f0, src_coded_sp, src_ap, tar_coded_sp

    def __getitem__(self, index):
        """get a sample

        Args:
            index (int): index of the entries

        Returns:
            dict: sample::

            {
                "src_coded_sp": src_coded_sp,  # sp_features x T
                "src_speaker": src_one_hot,
                "src_f0": src_f0,  # T
                "src_ap": src_ap,  # sp_features*fftsize//2+1
                "tar_coded_sp": tar_coded_sp,  # sp_features x T
                "tar_speaker": tar_one_hot,
                "src_id": self.speakers_dict[src_speaker],
                "tar_id": self.speakers_dict[tar_speaker],
                "src_wav_filename": src_wav_filename,
                "input_length": src_coded_sp.shape[1],
            }
        """
        src_wav_filename, src_speaker, tar_wav_filename, tar_speaker = \
            self.entries[index]
        if self.hparams.enable_load_from_disk:
            src_f0, src_coded_sp, src_ap, tar_coded_sp = self.load_from_disk(
                src_wav_filename, tar_wav_filename)
        else:
            src_wav, _ = librosa.load(src_wav_filename, sr=self.fs, mono=True, dtype=np.float64)
            tar_wav, _ = librosa.load(tar_wav_filename, sr=self.fs, mono=True, dtype=np.float64)
            src_f0, src_sp, src_ap, src_coded_sp = self.world_feature_extract(src_wav)
            tar_f0, tar_sp, tar_ap, tar_coded_sp = self.world_feature_extract(tar_wav)

        src_coded_sp = self.feature_normalizer(src_coded_sp, src_speaker)
        tar_coded_sp = self.feature_normalizer(tar_coded_sp, tar_speaker)
        src_coded_sp = tf.expand_dims(tf.transpose(src_coded_sp, [1, 0]), -1)
        tar_coded_sp = tf.expand_dims(tf.transpose(tar_coded_sp, [1, 0]), -1)

        # Convert the speaker names to one-hot
        src_one_hot = self.spk_one_hot[src_speaker]
        tar_one_hot = self.spk_one_hot[tar_speaker]
        src_wav_filename = os.path.split(src_wav_filename)[1]#.split("-")[1]

        return {
            "src_coded_sp": src_coded_sp,  # sp_features x T
            "src_speaker": src_one_hot,
            "src_f0": src_f0,  # T
            "src_ap": src_ap,  # sp_features*fftsize//2+1
            "tar_coded_sp": tar_coded_sp,  # sp_features x T
            "tar_speaker": tar_one_hot,
            "src_id": self.speakers_dict[src_speaker],
            "tar_id": self.speakers_dict[tar_speaker],
            "src_wav_filename": src_wav_filename,
            "input_length": src_coded_sp.shape[1],
        }

    def speaker_list(self):
        """ return the speaker list """
        return self.speakers

    @property
    def sample_type(self):
        """:obj:`@property`

        Returns:
            dict: sample_type of the dataset::

            {
                "src_coded_sp": tf.float32,
                "tar_coded_sp": tf.float32,
                "src_speaker": tf.float32,
                "tar_speaker": tf.float32,
                "src_f0": tf.float32,
                "src_ap": tf.float32,
                "src_id": tf.int32,
                "tar_id": tf.int32,
                "src_wav_filename": tf.string,
                "input_length": tf.int32,
            }
        """
        return {
            "src_coded_sp": tf.float32,
            "tar_coded_sp": tf.float32,
            "src_speaker": tf.float32,
            "tar_speaker": tf.float32,
            "src_f0": tf.float32,
            "src_ap": tf.float32,
            "src_id": tf.int32,
            "tar_id": tf.int32,
            "src_wav_filename": tf.string,
            "input_length": tf.int32,
        }

    @property
    def sample_shape(self):
        """:obj:`@property`

        Returns:
            dict: sample_shape of the dataset::

            {
                "src_coded_sp": tf.TensorShape(
                    [sp_dim, None, None]
                ),
                "tar_coded_sp": tf.TensorShape(
                    [sp_dim, None, None]
                ),
                "src_speaker": tf.TensorShape([None, spk_num]),
                "tar_speaker": tf.TensorShape([None, spk_num]),
                "src_f0": tf.TensorShape([None]),       # T
                "src_ap": tf.TensorShape([None, inc]),  # T*fftsize//2+1
                "src_id": tf.TensorShape([]),
                "tar_id": tf.TensorShape([]),
                "src_wav_filename": tf.TensorShape([]),
                "input_length": tf.TensorShape([]),
            }
        """
        sp_dim = self.sp_dim
        spk_num = self.spk_num
        inc = self.fft_size / 2 + 1
        return {
            "src_coded_sp": tf.TensorShape(
                [sp_dim, None, None]
            ),
            "tar_coded_sp": tf.TensorShape(
                [sp_dim, None, None]
            ),
            "src_speaker": tf.TensorShape([None, spk_num]),
            "tar_speaker": tf.TensorShape([None, spk_num]),
            "src_f0": tf.TensorShape([None]),       # T
            "src_ap": tf.TensorShape([None, inc]),  # T*fftsize//2+1
            "src_id": tf.TensorShape([]),
            "tar_id": tf.TensorShape([]),
            "src_wav_filename": tf.TensorShape([]),
            "input_length": tf.TensorShape([]),
        }

    @property
    def sample_signature(self):
        """:obj:`@property`

        Returns:
            dict: sample_signature of the dataset::

            {
                "src_coded_sp": tf.TensorSpec(
                    shape=(None, sp_dim, None, None), dtype=tf.float32),
                "tar_coded_sp": tf.TensorSpec(
                    shape=(None, sp_dim, None, None), dtype=tf.float32),
                "src_speaker": tf.TensorSpec(shape=(None, None, spk_num), dtype=tf.float32),
                "tar_speaker": tf.TensorSpec(shape=(None, None, spk_num), dtype=tf.float32),
                "src_f0": tf.TensorSpec(shape=(None, None), dtype=tf.float32),       # T
                "src_ap": tf.TensorSpec(shape=(None, None, inc), dtype=tf.float32),  # T*fftsize//2+1
                "src_id": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "tar_id": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "src_wav_filename": tf.TensorSpec(shape=(None), dtype=tf.string),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
            }
        """
        sp_dim = self.sp_dim
        spk_num = self.spk_num
        inc = self.fft_size / 2 + 1
        return (
            {
                "src_coded_sp": tf.TensorSpec(
                    shape=(None, sp_dim, None, None), dtype=tf.float32),
                "tar_coded_sp": tf.TensorSpec(
                    shape=(None, sp_dim, None, None), dtype=tf.float32),
                "src_speaker": tf.TensorSpec(shape=(None, None, spk_num), dtype=tf.float32),
                "tar_speaker": tf.TensorSpec(shape=(None, None, spk_num), dtype=tf.float32),
                "src_f0": tf.TensorSpec(shape=(None, None), dtype=tf.float32),      # T
                "src_ap": tf.TensorSpec(shape=(None, None, inc), dtype=tf.float32), # T*fftsize//2+1
                "src_id": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "tar_id": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "src_wav_filename": tf.TensorSpec(shape=(None), dtype=tf.string),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
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
            self.feature_normalizer.compute_world_cmvn(
                self.enable_load_from_disk, self.entries_person_wavs, \
                self.sp_dim, self.fft_size, self.fs, self.speakers)
        self.feature_normalizer.save_cmvn(
            ["speaker", "codedsp_mean", "codedsp_var", "f0_mean", "f0_var"])
        return self
