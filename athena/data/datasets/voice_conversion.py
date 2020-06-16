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
        

    def reload_config(self, config):
        """ reload the config """
        if config is not None:
            self.hparams.override_from_dict(config)

    def preprocess_data(self, file_path):
        """Generate a list of tuples (wav_filename, wav_length_ms, speaker)."""
        logging.info("Loading data from {}".format(file_path))

        with open(file_path, "r", encoding='utf-8') as file:
            lines = file.read().splitlines()
        random.shuffle(lines)  #shuffle the wav file
        headers = lines[0]
        lines = lines[1:]

        self.entries = []
        for line in lines:
            items = line.split("\t")
            # 需要读入两组音频
            wav_filename, speaker = items[0], items[1]
            if 'speaker' in headers.split("\t"):
                speaker = items[1]
            self.entries.append(tuple([wav_filename, speaker]))
        # self.entries.sort(key=lambda item: float(item[1]))   # 按语音长度排序

        self.speakers = []
        for  _, speaker in self.entries:
            if speaker not in self.speakers:
                self.speakers.append(speaker)
                # print("speaker:::",speaker)

        # self.filter_sample_by_input_length()
        return self

    def load_csv(self, file_path):
        """ load csv file """
        return self.preprocess_data(file_path)

    def __getitem__(self, index):
        audio_file, speaker = self.entries[index]
        
        feat = self.audio_featurizer(audio_file)
        # feat = self.feature_normalizer(feat, speaker)

        # fs=16000
        # sp_dim=36
        # spk_num = len(self.speakers)
        # wav, _ = librosa.load(audio_file, sr=fs, mono=True, dtype=np.float64)
        # fft_size=1024
        # # 对每个说话人取10个样本计算统计量
        # for i in range(spk_num):
        #     coded_sps,f0s = [],[]
        #     for j in range(10):
        #         audio_file, speaker = self.entries[i*70+j]
        #         wav, _ = librosa.load(audio_file, sr=fs, mono=True, dtype=np.float64)
        #         f0, timeaxis = pyworld.harvest(wav, fs)
        #         # CheapTrick harmonic spectral envelope estimation algorithm.
        #         sp = pyworld.cheaptrick(wav, f0, timeaxis, fs, fft_size=fft_size)
        #         # D4C aperiodicity estimation algorithm.
        #         ap = pyworld.d4c(wav, f0, timeaxis, fs, fft_size=fft_size)
        #         # feature reduction nxdim
        #         coded_sp = pyworld.code_spectral_envelope(sp, fs, sp_dim)  #压缩sp纬度为36
        #         # log
        #         coded_sp = coded_sp.T  # dim x n
        input_data = feat
        output_data = tf.reshape(
            feat, [-1, self.audio_featurizer.dim * self.audio_featurizer.num_channels]
        )

        return {
            "input": input_data,
            "input_length": input_data.shape[0],
            "output": output_data,
            "output_length": output_data.shape[0],
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
            "input": tf.float32,
            "input_length": tf.int32,
            "output": tf.float32,
            "output_length": tf.int32,
        }

    @property
    def sample_shape(self):
        return {
            "input": tf.TensorShape(
                [None, self.audio_featurizer.dim, self.audio_featurizer.num_channels]
            ),
            "input_length": tf.TensorShape([]),
            "output": tf.TensorShape([None, None]),
            "output_length": tf.TensorShape([]),
        }

    @property
    def sample_signature(self):
        return (
            {
                "input": tf.TensorSpec(
                    shape=(None, None, None, None), dtype=tf.float32
                ),
                "input_length": tf.TensorSpec(shape=([None]), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                "output_length": tf.TensorSpec(shape=([None]), dtype=tf.int32),
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
        # if os.path.exists(self.hparams.cmvn_file):
        #     return self

        # 计算world特征的均值方差   entries存放路劲和说话人
        spk_num = len(self.speakers)-1
        cmvns = []
        fs ,sp_dim =16000,36
        fft_size=1024
        # 对每个说话人取10个样本计算统计量
        for i in range(spk_num):
            coded_sps,f0s = [],[]
            for j in range(2):
                print("line_num",i*70+j)
                audio_file, speaker = self.entries[i*70+j]
                wav, _ = librosa.load(audio_file, sr=fs, mono=True, dtype=np.float64)
                f0, timeaxis = pyworld.harvest(wav, fs)
                # CheapTrick harmonic spectral envelope estimation algorithm.
                sp = pyworld.cheaptrick(wav, f0, timeaxis, fs, fft_size=fft_size)
                # D4C aperiodicity estimation algorithm.
                ap = pyworld.d4c(wav, f0, timeaxis, fs, fft_size=fft_size)
                # feature reduction nxdim
                coded_sp = pyworld.code_spectral_envelope(sp, fs, sp_dim)  #压缩sp纬度为36
                # log
                coded_sp = coded_sp.T  # dim x n
                coded_sps.append(coded_sp)
                f0_ = np.reshape(f0, [-1, 1])
                print(f'f0 shape: {f0_.shape}')
                f0s.append(f0_)

            # 计算当前说话人的统计数据
            coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
            coded_sps_mean = list(np.mean(coded_sps_concatenated, axis=1, keepdims=False))
            coded_sps_std = list(np.std(coded_sps_concatenated, axis=1, keepdims=False))
            log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
            log_f0s_mean = log_f0s_concatenated.mean()
            log_f0s_std = log_f0s_concatenated.std()
            # cmvn_dict[speaker] = (list(mean_i.numpy()), list(variance_i.numpy()))
            # print("show::",speaker, coded_sps_mean , coded_sps_std,log_f0s_mean, log_f0s_std)
            cmvns.append(
                    (speaker, coded_sps_mean , coded_sps_std,log_f0s_mean, log_f0s_std))

        df = pandas.DataFrame(data=cmvns, columns=["speaker", "codedsp_mean", "codedsp_var","f0_mean", "f0_var"])
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