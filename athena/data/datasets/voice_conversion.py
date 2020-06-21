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
import os
from absl import logging
import tensorflow as tf
from athena.transform import AudioFeaturizer
from ...utils.hparam import register_and_parse_hparams
from ..feature_normalizer import FeatureNormalizer2
from .base import BaseDatasetBuilder
import librosa
import pyworld
import numpy as np
import pandas
import random
import glob


class VoiceConversionDatasetBuilder(BaseDatasetBuilder):
    """ SpeechDatasetBuilder

    Args:
        for __init__(self, config=None)

    Config:
        feature_config: the config file for feature extractor, default={'type':'Fbank'}
        data_csv: the path for original LDC HKUST,
            default='/tmp-data/dataset/opensource/hkust/train.csv'
        force_process: force process, if force_process=True, we will re-process the dataset,
            if False, we will process only if the out_path is empty. default=False

    Interfaces::
        __len__(self): return the number of data samples

        @property:
        sample_shape:
            {"input": tf.TensorShape([None, self.audio_featurizer.dim,
                                  self.audio_featurizer.num_channels]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None, self.audio_featurizer.dim *
                                  self.audio_featurizer.num_channels]),}
    """
    default_config = {
        "audio_config": {"type": "World"},
        "cmvn_file": None,
        "input_length_range": [20, 50000],
        "data_csv": None,
        "num_cmvn_workers": 1
    }

    def __init__(self, config=None):
        super().__init__()
        # hparams
        self.hparams = register_and_parse_hparams(
            self.default_config, config, cls=self.__class__)
        logging.info("hparams: {}".format(self.hparams))
        self.audio_featurizer = AudioFeaturizer(self.hparams.audio_config)
        # self.feature_normalizer = FeatureNormalizer2(self.hparams.cmvn_file)
        if self.hparams.data_csv is not None:
            self.load_csv(self.hparams.data_csv)

        # creat one-hot speaker embedding
        self.spker_num = len(self.speakers)
        self.spk_one_hot = {}
        onehot_arr = np.eye(self.spker_num)
        spk_count = 0
        for name in self.speakers:
            self.spk_one_hot[name] = onehot_arr[:, spk_count]
            spk_count = spk_count + 1

    def load_csv(self, file_path):
        """Generate a list of tuples (src_wav_filename, src_speaker,
                                      tar_wav_filename, tar_speaker)."""
        logging.info("Loading data from {}".format(file_path))
        with open(file_path, "r", encoding='utf-8') as file:
            lines = file.read().splitlines()
        lines = lines[1:]
        lines.shuffle(lines)  # shuffle the wav file

        self.entries = []
        for line in lines:
            items = line.split("\t")
            src_wav_filename, src_speaker = items[0], items[1]
            for line2 in lines:
                items2 = line2.split("\t")
                tar_wav_filename, tar_speaker = items2[0], items2[1]
                if src_speaker != tar_speaker:
                    self.entries.append(tuple([src_wav_filename, src_speaker,
                                               tar_wav_filename, tar_speaker]))

        self.speakers = []
        for _, speaker, _  in self.entries:
            if speaker not in self.speakers:
                self.speakers.append(speaker)

        return self

    def reload_config(self, config):
        """ reload the config """
        if config is not None:
            self.hparams.override_from_dict(config)

    def pad_wav_to_get_fixed_frames(x: np.ndarray, y: np.ndarray):
        wav_len_x = len(x)
        wav_len_y = len(y)
        if wav_len_x > wav_len_y:
            need_pad = wav_len_x - wav_len_y
            tempx = y.tolist()
            tempx.extend(y[:need_pad])
            y = tempx[:-1]
            paded_len = wav_len_x
        else:
            need_pad = wav_len_y - wav_len_x
            tempx = x.tolist()
            tempx.extend(x[:need_pad])
            x = tempx[:-1]
            paded_len = wav_len_y

        return np.asarray(x, dtype=np.float), np.asarray(y, dtype=np.float), paded_len

    def __getitem__(self, index):
        src_wav_filename, src_speaker, tar_wav_filename, tar_speaker = \
            self.entries[index]

        srcwav, _ = librosa.load(src_wav_filename, sr=self.hparams.fs, mono=True, dtype=np.float32)
        tarwav, _ = librosa.load(tar_wav_filename, sr=self.hparams.fs, mono=True, dtype=np.float32)
        src_wav, tar_wav, wav_len = self.pad_wav_to_get_fixed_frames(srcwav, tarwav)

        # extract the world features of the source speaker
        src_f0, timeaxis = pyworld.harvest(src_wav, self.hparams.fs)
        src_sp = pyworld.cheaptrick(src_wav, src_f0, timeaxis, self.hparams.fs, fft_size=self.hparams.fft_size)
        src_ap = pyworld.d4c(src_wav, src_f0, timeaxis, self.hparams.fs, fft_size=self.hparams.fft_size)
        src_coded_sp = pyworld.code_spectral_envelope(src_sp, self.hparams.fs, self.hparams.coded_sp_dim)
        src_coded_sp = src_coded_sp.T  # dim x n

        # extract the world features of the target speaker
        tar_f0, timeaxis = pyworld.harvest(tar_wav, self.hparams.fs)
        tar_sp = pyworld.cheaptrick(tar_wav, tar_f0, timeaxis, self.hparams.fs, fft_size=self.hparams.fft_size)
        tar_ap = pyworld.d4c(tar_wav, tar_f0, timeaxis, self.hparams.fs, fft_size=self.hparams.fft_size)
        tar_coded_sp = pyworld.code_spectral_envelope(tar_sp, self.hparams.fs, self.hparams.coded_sp_dim)
        tar_coded_sp = tar_coded_sp.T  # dim x n

        src_one_hot = self.spk_one_hot[src_speaker]
        tar_one_hot = self.spk_one_hot[tar_speaker]

        return {
            'src_coded_sp': src_coded_sp,  # dim * n
            'tar_coded_sp': tar_coded_sp,  # dim * n
            "src_speaker": src_one_hot,
            "tar_speaker": tar_one_hot,
            "input_length": src_coded_sp.shape[1],
            'src_f0': src_f0,  # n
            'src_ap': src_ap,  # n*fftsize//2+1
            'src_sp': src_sp,  # n*fftsize//2+1
            'tar_f0': tar_f0,  # n
            'tar_ap': tar_ap,  # n*fftsize//2+1
            'tar_sp': tar_sp,  # n*fftsize//2+1
            
        }

    def __len__(self):
        """ return the number of data samples """
        return len(self.entries)

    @property
    def num_class(self):
        """ return the max_index of the vocabulary """
        target_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        return target_dim

    @property
    def speaker_list(self):
        """ return the speaker list """
        return self.speakers

    @property
    def audio_featurizer_func(self):
        """ return the audio_featurizer function """
        return self.audio_featurizer

    @property
    def sample_type(self):
        return {
            "src_coded_sp": tf.float32,
            "tar_coded_sp": tf.float32,
            "src_speaker": tf.float32,
            "tar_speaker": tf.float32,
            "input_length": tf.int32,
            "src_f0": tf.float32,# n
            "src_ap": tf.float32,  # n*fftsize//2+1
            "src_sp": tf.float32,  # n*fftsize//2+1
            "tar_f0": tf.float32, # n
            "tar_ap": tf.float32,  # n*fftsize//2+1
            "tar_sp": tf.float32,  # n*fftsize//2+1
        }

    @property
    def sample_shape(self):
        sp_dim = self.hparams.coded_sp_dim
        nc = self.audio_featurizer.num_channels
        spk_num = self.hparams.speaker_num
        inc = self.hparams.fft_size/2+1
        return {
            #  num_features, frames, 1
            "src_coded_sp": tf.TensorShape(
                [None, sp_dim, nc]
            ),
            "tar_coded_sp": tf.TensorShape(
                [None, sp_dim, nc]
            ),
            "src_speaker": tf.TensorShape([spk_num]),
            "tar_speaker": tf.TensorShape([spk_num]),
            "input_length": tf.TensorShape([]),
            "src_f0": tf.TensorShape([None]),  # n
            "src_ap": tf.TensorShape([None,inc]),  # n*fftsize//2+1
            "src_sp": tf.TensorShape([None,inc]),  # n*fftsize//2+1
            "tar_f0": tf.TensorShape([None]),  # n
            "tar_ap": tf.TensorShape([None,inc]),  # n*fftsize//2+1
            "tar_sp": tf.TensorShape([None,inc]),  # n*fftsize//2+1
            
        }

    @property
    def sample_signature(self):
        sp_dim = self.hparams.coded_sp_dim
        nc = self.audio_featurizer.num_channels
        spk_num = self.hparams.speaker_num
        inc = self.hparams.fft_size / 2 + 1
        return (
            {
                "src_coded_sp": tf.TensorShape(
                    [None, None,sp_dim, nc]
                ),
                "tar_coded_sp": tf.TensorShape(
                    [None, None,sp_dim, nc]
                ),
                "src_speaker": tf.TensorShape([None,spk_num]),
                "tar_speaker": tf.TensorShape([None,spk_num]),
                "input_length": tf.TensorShape([None]),
                "src_f0": tf.TensorShape([None,None]),  # n
                "src_ap": tf.TensorShape([None,None, inc]),  # n*fftsize//2+1
                "src_sp": tf.TensorShape([None,None, inc]),  # n*fftsize//2+1
                "tar_f0": tf.TensorShape([None,None]),  # n
                "tar_ap": tf.TensorShape([None,None, inc]),  # n*fftsize//2+1
                "tar_sp": tf.TensorShape([None,None, inc]),  # n*fftsize//2+1
            },
        )

    def filter_sample_by_input_length(self):
        """filter samples by input length

        The length of filterd samples will be in [min_length, max_length)

        Args:
            self.hparams.input_length_range = [min_len, max_len]
            min_len: the minimal length(ms)
            max_len: the maximal length(ms)
        returns:
            entries: a filtered list of tuples
            (wav_filename, wav_len, speaker)
        """
        min_len = self.hparams.input_length_range[0]
        max_len = self.hparams.input_length_range[1]
        filter_entries = []
        for wav_filename, wav_len, speaker in self.entries:
            if int(wav_len) in range(min_len, max_len):
                filter_entries.append(tuple([wav_filename, wav_len, speaker]))
        self.entries = filter_entries

    def compute_cmvn_if_necessary(self, is_necessary=True):
        """ compute cmvn file
        """
        if not is_necessary:
            return self
        if os.path.exists(self.hparams.cmvn_file):
            return self

        spk_num = len(self.speakers)
        cmvns = []
        sp_dim = self.hparams.coded_sp_dim
        fft_size = self.hparams.fft_size

        with open(self.hparams.data_csv, "r", encoding='utf-8') as file:
            lines = file.read().splitlines()
        lines = lines[1:]
        self.entries_person = []
        for line in lines:
            items = line.split("\t")
            wav_filename, speaker = items[0], items[1]
            self.entries_person.append(tuple([wav_filename, speaker]))

        for i in range(spk_num):
            coded_sps, f0s = [], []
            for j in range(self.hparams.people_sentences):
                audio_file, speaker = self.entries_person[i * self.hparams.people_sentences + j]
                wav, _ = librosa.load(audio_file, sr=self.hparams.fs, mono=True, dtype=np.float64)
                f0, timeaxis = pyworld.harvest(wav, self.hparams.fs)
                # CheapTrick harmonic spectral envelope estimation algorithm.
                sp = pyworld.cheaptrick(wav, f0, timeaxis, self.hparams.fs, fft_size=fft_size)
                # feature reduction nxdim
                coded_sp = pyworld.code_spectral_envelope(sp, self.hparams.fs, sp_dim)
                coded_sp = coded_sp.T  # dim x n
                coded_sps.append(coded_sp)
                f0_ = np.reshape(f0, [-1, 1])
                f0s.append(f0_)

            coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
            coded_sps_mean = list(np.mean(coded_sps_concatenated, axis=1, keepdims=False))
            coded_sps_std = list(np.std(coded_sps_concatenated, axis=1, keepdims=False))
            log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
            log_f0s_mean = log_f0s_concatenated.mean()
            log_f0s_std = log_f0s_concatenated.std()
            cmvns.append(
                (speaker, coded_sps_mean, coded_sps_std, log_f0s_mean, log_f0s_std))

        df = pandas.DataFrame(data=cmvns, columns=["speaker", "codedsp_mean", "codedsp_var", "f0_mean", "f0_var"])
        df.to_csv(self.hparams.cmvn_file, index=False, sep="\t")

        # feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        # with tf.device("/cpu:0"):
        #     self.feature_normalizer.compute_cmvn(
        #         self.entries, self.speakers, self.audio_featurizer, feature_dim, self.hparams.num_cmvn_workers
        #     )
        # self.feature_normalizer.save_cmvn()
        return self


if __name__ == "__main__":
    import json
    from absl import logging
    from athena.main import parse_config, SUPPORTED_DATASET_BUILDER

    jsonfile, csv_file = "examples/vc/vcc2020/configs/VC.json", "examples/vc/vcc2020/data/train.csv"
    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)
    if "speed_permutation" in p.trainset_config:
        p.dataset_config['speed_permutation'] = [1.0]
    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.trainset_config)
    dataset_builder.load_csv(csv_file).compute_cmvn_if_necessary(True)