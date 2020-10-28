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
import numpy as np
from absl import logging
import random
import tensorflow as tf
from athena.transform import AudioFeaturizer
from ...utils.hparam import register_and_parse_hparams
from ..feature_normalizer import FeatureNormalizer
from .base import BaseDatasetBuilder
from .preprocess import SpecAugment


class SpeechAugmentDatasetBuilder(BaseDatasetBuilder):
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
        "audio_config": {"type": "Fbank"},
        "cmvn_file": None,
        "input_length_range": [20, 50000],
        "data_csv": None,
        "spec_augment": {
            "time_warping": 0,
            "time_masking": 0,
            "frequency_masking": 0,
            "mask_cols": 0,
            "mask_type": "mean"
        },
        "speed_permutation": [0.9, 1.0, 1.1]
    }

    def __init__(self, config=None):
        super().__init__()
        # hparams
        self.hparams = register_and_parse_hparams(
            self.default_config, config, cls=self.__class__)
        logging.info("hparams: {}".format(self.hparams))

        self.audio_featurizer = AudioFeaturizer(self.hparams.audio_config)
        self.feature_normalizer = FeatureNormalizer(self.hparams.cmvn_file)
        self.speech_augment = SpecAugment(self.hparams.spec_augment)
        if self.hparams.data_csv is not None:
            self.load_csv(self.hparams.data_csv)
        self.aug_rate = 0.8
        #self.process_speed_path = "/nfs/cold_project/caomiao/data/hkust/train"
        self.process_speed_path = "/nfs/cold_project/datasets/opensource_data/hkust_feature"
        #self.process_speed_path = "/nfs/cold_project/datasets/opensource_data/kehu_feature"

    def reload_config(self, config):
        """ reload the config """
        if config is not None:
            self.hparams.override_from_dict(config)

    def preprocess_data(self, file_path):
        """Generate a list of tuples (wav_filename, wav_length_ms, speaker)."""
        logging.info("Loading data from {}".format(file_path))
        with open(file_path, "r", encoding='utf-8') as file:
            lines = file.read().splitlines()
        headers = lines[0]
        lines = lines[1:]

        self.entries = []
        for line in lines:
            items = line.split("\t")
            wav_filename, length, speaker = items[0], items[1], 'global'
            if 'speaker' in headers.split("\t"):
                speaker = items[-1]
            self.entries.append(tuple([wav_filename, length, speaker]))
        self.entries.sort(key=lambda item: float(item[1]), reverse=True)

        self.speakers = []
        for _, _, speaker in self.entries:
            if speaker not in self.speakers:
                self.speakers.append(speaker)

        self.filter_sample_by_input_length()
        return self

    def load_csv(self, file_path):
        """ load csv file """
        return self.preprocess_data(file_path)

    def __getitem__(self, index):
        feats = []
        audio_file, _, speaker = self.entries[index]

        #feat = self.audio_featurizer(audio_file)
        #if random.uniform(0, 1) < self.aug_rate:
        feat_id = os.path.basename(audio_file)
        feat_id += '.npy'
        try:
            if "train-other-500-wav" in audio_file or "train-clean-360-wav" in audio_file or "train-clean-100-wav"in audio_file:
                ori = '/nfs/project/jiangdongwei/librispeech-processed/train/numpy/' + feat_id
                aug = '/nfs/project/jiangdongwei/librispeech-processed/train_aug/numpy/' + feat_id
            else:
                ori = '/nfs/project/jiangdongwei/librispeech-processed/dev/numpy/' + feat_id
                aug = '/nfs/project/jiangdongwei/librispeech-processed/dev_aug/numpy/' + feat_id
            feat_ori = np.load(ori)
            feat_aug = np.load(aug)
        except:
            feat_ori = np.zeros([400, 40, 1], dtype=np.float32)
            feat_aug = np.zeros([400, 40, 1], dtype=np.float32)
        feat_ori = tf.convert_to_tensor(feat_ori)
        feat_aug = tf.convert_to_tensor(feat_aug)
        #feat = self.audio_featurizer(audio_file, speed=random.choice(self.hparams.speed_permutation))
        feat_ori = self.feature_normalizer(feat_ori, speaker)
        feat_aug = self.feature_normalizer(feat_aug, speaker)
        #if random.uniform(0, 1) < self.aug_rate:
        #feat = self.speech_augment(feat)
        feats.append(feat_ori)
        feats.append(feat_aug)

        return {
            "input": feats[0],
            "input_length": feats[0].shape[0],
            "output": feats[1],
            "output_length": feats[1].shape[0],
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
            "output": tf.TensorShape([None, self.audio_featurizer.dim, self.audio_featurizer.num_channels]),
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
                "output": tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
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
        if os.path.exists(self.hparams.cmvn_file):
            return self
        feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        with tf.device("/cpu:0"):
            self.feature_normalizer.compute_cmvn(
                self.entries, self.speakers, self.audio_featurizer, feature_dim
            )
        self.feature_normalizer.save_cmvn()
        return self
