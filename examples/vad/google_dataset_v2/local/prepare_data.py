# coding=utf-8
# Copyright (C) ATHENA AUTHORS
# All rights reserved.
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
# pylint: disable=invalid-name

#!/usr/bin/python
# -*- coding: utf-8 -*-

""" google_dataset_v2 dataset """

import os
import sys
import glob
import json
import kaldiio
import librosa
import random
import numpy as np
import tensorflow as tf

from absl import logging
from athena.transform import feats
from athena.transform import AudioFeaturizer
from sklearn.model_selection import train_test_split

sampling_rate = 16000
labels_vocab = ['BACKGROUND', 'SPEECH']
audio_config = {"type": "Fbank", "filterbank_channel_count":80}

def split_train_val_test(dataset_dir, dev_size=0.1, test_size=0.1):
    wav_file_list = []
    print(dataset_dir)
    for name in os.listdir(dataset_dir):
        sub_dir = os.path.join(dataset_dir, name)
        if not os.path.isdir(sub_dir):
            continue
        sub_file_list = glob.glob(sub_dir + '/*.wav')
        # skip _background_noise_ in google_speech_recognition_v2
        if sub_dir.split("/")[-1] != "_background_noise_" and sub_dir.split("/")[-1] != "real_rirs_isotropic_noises":
            wav_file_list.extend(sub_file_list)
        else:
            continue

    # spliting train/dev/test list
    wav_train, wav_dev = train_test_split(wav_file_list, test_size=dev_size, random_state=1)
    wav_train, wav_test = train_test_split(wav_train, test_size=test_size, random_state=1)

    logging.info(f'DATA DIR: {dataset_dir}')
    logging.info(f'DATA Overall: {len(wav_file_list)}, Train: {len(wav_train)}, Validatoin: {len(wav_dev)}, Test: {len(wav_test)}')
    return wav_train, wav_dev, wav_test

def extract_feat(wav_list,
                 label,
                 start = 0.0,
                 end = None):
    feat_dict = {}
    label_dict = {}
    # config for extracting feats
    read_wav = getattr(feats, "ReadWav").params(audio_config).instantiate()
    audio_featurizer = AudioFeaturizer(audio_config)
    # processing feats
    for cur_wav in wav_list:
        # extracting feats
        audio_data, sr = read_wav(cur_wav)
        audio_data_start_position = tf.math.round(start * tf.cast(sr, dtype=tf.float32))
        audio_data_end_position = audio_data.shape[0] 
        if end != None and end < audio_data.shape[0] / sr:
            audio_data_end_position = tf.math.round(end * tf.cast(sr, dtype=tf.float32))
        audio_data_start_position = tf.cast(audio_data_start_position, dtype=tf.int32)
        audio_data_end_position = tf.cast(audio_data_end_position, dtype=tf.int32)
        cur_feats = audio_featurizer(audio_data[audio_data_start_position:audio_data_end_position], sr)
        # storage features offline
        cur_key = cur_wav
        cur_feats = np.reshape(
                 cur_feats, (-1, audio_config["filterbank_channel_count"]))
        cur_labels = [labels_vocab.index(label) for _ in range(cur_feats.shape[0])] 
        cur_labels = np.array(cur_labels, dtype='int32')
        feat_dict[cur_key] = cur_feats
        label_dict[cur_key] = cur_labels
    return feat_dict, label_dict

def select_dict_by_frames_num(num_frames, input_dict):
    selected_dict = {}
    selected_frames_num = 0
    for key in input_dict:
        cur_frames_num = input_dict[key].shape[0]
        if selected_frames_num + cur_frames_num > num_frames:
            continue
        selected_dict[key] = input_dict[key]
        selected_frames_num += cur_frames_num
    return selected_dict

def rebalance_data(feat_dict_1, label_dict_1, feat_dict_2, label_dict_2):
    utt2num_frames_1 = sum([feat_dict_1[key].shape[0] for key in feat_dict_1])
    utt2num_frames_2 = sum([feat_dict_2[key].shape[0] for key in feat_dict_2])
    if utt2num_frames_1 < utt2num_frames_2:
        min_utt2num_frames = utt2num_frames_1
        feat_dict_2 = select_dict_by_frames_num(min_utt2num_frames, feat_dict_2)
        label_dict_2 = select_dict_by_frames_num(min_utt2num_frames, label_dict_2)
    else:
        min_utt2num_frames = utt2num_frames_2
        feat_dict_1 = select_dict_by_frames_num(min_utt2num_frames, feat_dict_1)
        label_dict_1 = select_dict_by_frames_num(min_utt2num_frames, label_dict_1)
    feat_dict = feat_dict_1.copy()
    feat_dict.update(feat_dict_2)
    label_dict = label_dict_1.copy()
    label_dict.update(label_dict_2)
    return feat_dict, label_dict

def train_scp_post_process(feats_dict, labels_dict, split_len=63):
    split_feats_dict, split_labels_dict = {}, {}
    processed_feat_dict, processed_labels_dict = {}, {}
    for key in feats_dict.keys():
        cur_feats = feats_dict[key]
        cur_labels = labels_dict[key]
        cur_len = cur_feats.shape[0]
        for pos in range(0, cur_len, split_len):
            tmp_feats = cur_feats[pos:pos + split_len]
            tmp_labels = cur_labels[pos:pos + split_len]
            tmp_key = key + '_' + str(pos // split_len)
            split_feats_dict[tmp_key] = tmp_feats
            split_labels_dict[tmp_key] = tmp_labels
    split_key_list = list(split_labels_dict.keys())
    random.shuffle(split_key_list)
    for key in split_key_list:
        processed_feat_dict[key] = split_feats_dict[key]
        processed_labels_dict[key] = split_labels_dict[key]
    return processed_feat_dict, processed_labels_dict

def prepare_scp(out_dir, data_type, speech_wav_list, background_wav_list):
    speech_feat_dict, speech_label_dict = extract_feat(speech_wav_list, 'SPEECH', start=0.2, end=0.8)
    background_feat_dict, background_label_dict = extract_feat(background_wav_list, 'BACKGROUND', start=0, end=None)
    feat_dict, label_dict = rebalance_data(speech_feat_dict, speech_label_dict, background_feat_dict, background_label_dict) 
    if data_type in ['train', 'dev']:
        feat_dict, label_dict = train_scp_post_process(feat_dict, label_dict, split_len=63)
    kaldiio.save_ark(os.path.join(out_dir, data_type, "feats.ark"),
                     feat_dict, scp=os.path.join(out_dir, data_type, "feats.scp"))
    kaldiio.save_ark(os.path.join(out_dir, data_type, "labels.ark"),
                     label_dict, scp=os.path.join(out_dir, data_type, "labels.scp"))

def process_data(speech_dataset_dir, background_dataset_dir, out_dir):
    speech_train, speech_dev, speech_test = split_train_val_test(speech_dataset_dir)
    background_train, background_dev, background_test = split_train_val_test(background_dataset_dir)
    prepare_scp(out_dir, 'train', speech_train, background_train)
    prepare_scp(out_dir, 'dev', speech_dev, background_dev)
    prepare_scp(out_dir, 'test', speech_test, background_test)

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        logging.info('Usage: python {} dataset_root_dir\n'
              '    speech_dataset_dir : speech dataset directory\n'
              '    background_dataset_dir : background dataset directory\n'
              '    out_dir : Athena working directory\n'.format(sys.argv[0]))
        exit(1)
    SPEECH_DATASET_DIR = sys.argv[1]
    BACKGROUND_DATASET_DIR = sys.argv[2]
    OUT_DIR = sys.argv[3]
    process_data(SPEECH_DATASET_DIR, BACKGROUND_DATASET_DIR, OUT_DIR)
