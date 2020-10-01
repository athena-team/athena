# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
# All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/python
# -*- coding: utf-8 -*-

""" VCC2018 dataset """

import os
import sys
import json
import glob
import numpy as np
import pyworld
from absl import logging
import librosa
from athena.main import parse_config

SUBSETS = ["train", "dev", "test"]

def wav_to_mcep_file(dataset, processed_filepath, sr, coded_sp_dim, fft_size):
    allwavs_cnt = len(glob.glob(f'{dataset}/*.wav'))
    print(f'Total {allwavs_cnt} audio files!')

    wavs_name = [x for x in os.listdir(dataset) if x.endswith('.wav') and not x.startswith('.')]
    for audio_name in wavs_name:
        wav_name = os.path.join(dataset, audio_name)
        audio_wav, _ = librosa.load(wav_name, sr=sr, mono=True, dtype=np.float64)
        # cal source audio feature
        f0, ap, coded_sp = cal_mcep(
            audio_wav, fs=sr, dim=coded_sp_dim, fft_size=fft_size)
        newname = audio_name.replace(".wav", ".npz")

        # save the dict as npz
        file_path_z = f'{processed_filepath}/{newname}'
        print(f'save file: {file_path_z}')
        np.savez(file_path_z, f0=f0, ap=ap, coded_sp=coded_sp)

def cal_mcep(wav_ori, fs, dim, fft_size):
    wav = wav_ori
    f0, timeaxis = pyworld.harvest(wav, fs)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs, fft_size=fft_size)
    ap = pyworld.d4c(wav, f0, timeaxis, fs, fft_size=fft_size)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)
    coded_sp = coded_sp.T  # dim x n
    return f0, ap, coded_sp

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        print('Usage: python {} dataset_dir output_dir\n'
              '    dataset_dir : directory contains VCC2020 dataset\n'
              '    output_dir  : Athena working directory'.format(sys.argv[0]))
        exit(1)
    DATASET_DIR = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    jsonfile = sys.argv[3]
    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)
    FEATURE_DIM = p.trainset_config["codedsp_dim"]
    SAMPLE_RATE = p.trainset_config["fs"]
    FFTSIZE = p.trainset_config["fft_size"]

    # we will generate a dev set of our own
    os.system('mkdir -p ' + os.path.join(DATASET_DIR, 'dev'))
    for dir in os.listdir(os.path.join(DATASET_DIR, 'train')):
        if not dir.startswith('V'):
            continue
        os.system('mkdir -p ' + os.path.join(DATASET_DIR, 'dev') + '/' + dir)
        file_list = os.listdir(os.path.join(DATASET_DIR, 'train') + '/' + dir)
        print(os.path.join(DATASET_DIR, 'train') + '/' + dir)
        print(file_list)
        # for each speaker, we take 5 samples for dev set
        for i in range(5):
            os.system('mv ' + os.path.join(DATASET_DIR, 'train') + '/' + dir + '/' + file_list[i] + ' ' + os.path.join(DATASET_DIR, 'dev') + '/' + dir)
    print('development set construction done!')

    for data in SUBSETS:
        input_dir = os.path.join(DATASET_DIR, data)
        output_dir = os.path.join(OUTPUT_DIR, data)
        spks = [x for x in os.listdir(input_dir)]
        for spk in spks:
            # there are some files, not dir, in vcc2018 data, also we don't want to deal with transcriptiosn
            if os.path.isfile(os.path.join(input_dir, spk)) or not spk.startswith('V'):
                continue
            input_spk = os.path.join(input_dir, spk)
            output_spk = os.path.join(output_dir, spk)
            print(input_spk, output_spk)
            if not os.path.exists(input_spk):
                os.makedirs(input_spk)
            if not os.path.exists(output_spk):
                os.makedirs(output_spk)
            wav_to_mcep_file(input_spk, processed_filepath=output_spk, sr=SAMPLE_RATE, \
                             coded_sp_dim=FEATURE_DIM, fft_size=FFTSIZE)
