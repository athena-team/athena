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
# pylint: disable=invalid-name
""" Feature Normalizer """
import os
import json
import time
import multiprocessing as mp
from multiprocessing import cpu_count
import pandas
import librosa
import pyworld
from absl import logging
import tensorflow as tf
import numpy as np
import tqdm


class FeatureNormalizer:
    """Feature Normalizer
    """

    def __init__(self, cmvn_file=None):
        super().__init__()
        self.cmvn_file = cmvn_file
        self.cmvn_dict = {}
        self.speakers = []
        if cmvn_file is not None:
            self.load_cmvn()

    def __call__(self, feat_data, speaker, reverse=False):
        return self.apply_cmvn(feat_data, speaker, reverse=reverse)

    def apply_cmvn(self, feat_data, speaker, reverse=False):
        """transform original feature to normalized feature
        """
        if speaker not in self.cmvn_dict:
            return feat_data
        mean = self.cmvn_dict[speaker][0]
        var = self.cmvn_dict[speaker][1]
        shape = feat_data.get_shape().as_list()[1:]
        mean = tf.reshape(tf.convert_to_tensor(mean, dtype=tf.float32), shape)
        var = tf.reshape(tf.convert_to_tensor(var, dtype=tf.float32), shape)
        if reverse:
            feat_data = feat_data * tf.sqrt(var) + mean
        else:
            feat_data = (feat_data - mean) / tf.sqrt(var)
        return feat_data

    def compute_cmvn(self, entries, speakers, featurizer, feature_dim, num_cmvn_workers=1):
        """compute cmvn for filtered entries
        """
        start = time.time()
        if num_cmvn_workers == 1:
            initial_mean, initial_var, total_num = self.compute_cmvn_by_chunk_for_all_speaker(
                feature_dim, speakers, featurizer, entries)

            for tar_speaker in speakers:
                # compute mean and var for all speaker
                if total_num[tar_speaker] == 0:
                    continue
                total_num_i = tf.cast(total_num[tar_speaker], tf.float32)
                mean_i = initial_mean[tar_speaker] / total_num_i
                variance_i = initial_var[tar_speaker] / total_num_i - tf.square(mean_i)
                self.cmvn_dict[tar_speaker] = (list(mean_i.numpy()), list(variance_i.numpy()))
        else:
            num_cmvn_workers = num_cmvn_workers if num_cmvn_workers else cpu_count()
            ctx = mp.get_context('spawn')
            m = ctx.Manager()
            args = []
            chunks = np.array_split(entries, num_cmvn_workers)
            for chunk in chunks:
                args.append((feature_dim, speakers, featurizer, chunk))
            p = ctx.Pool(num_cmvn_workers)
            # get results of all sub-process
            result_list = p.starmap(self.compute_cmvn_by_chunk_for_all_speaker, args)
            p.close()
            p.join()

            for tar_speaker in speakers:
                initial_mean_speaker = tf.Variable(tf.zeros([feature_dim], dtype=tf.float32))
                initial_var_speaker = tf.Variable(tf.zeros([feature_dim], dtype=tf.float32))
                total_num_speaker = tf.Variable(0, dtype=tf.int32)

                for chunk_initial_mean, chunk_initial_var, chunk_total_num in result_list:
                    initial_mean_speaker.assign_add(chunk_initial_mean[tar_speaker])
                    initial_var_speaker.assign_add(chunk_initial_var[tar_speaker])
                    total_num_speaker.assign_add(chunk_total_num[tar_speaker])
                # using the sums to compute mean and var for all speaker
                if total_num_speaker == 0:
                    continue
                total_num_i = tf.cast(total_num_speaker, tf.float32)
                mean_i = initial_mean_speaker / total_num_i
                variance_i = initial_var_speaker / total_num_i - tf.square(mean_i)
                self.cmvn_dict[tar_speaker] = (list(mean_i.numpy()), list(variance_i.numpy()))
        logging.info("finished compute cmvn, which cost %.4f s" % (time.time() - start))

    def compute_cmvn_by_chunk_for_all_speaker(self, feature_dim, speakers, featurizer, entries):
        """because of memory issue, we used incremental approximation for the calculation of cmvn
        """
        initial_mean_dict, initial_var_dict, total_num_dict = {}, {}, {}
        # speakers may be 'global' or a speaker list
        for tar_speaker in speakers:
            logging.info("processing %s from %s" % (tar_speaker, os.getpid()))
            # compute some sums of the corresponding chunk for per speaker
            initial_mean = tf.Variable(tf.zeros([feature_dim], dtype=tf.float32))
            initial_var = tf.Variable(tf.zeros([feature_dim], dtype=tf.float32))
            total_num = tf.Variable(0, dtype=tf.int32)

            for items in tqdm.tqdm(entries):
                audio_file, speaker = items[0], items[-1]
                if speaker != tar_speaker:
                    continue
                feat_data = featurizer(audio_file)
                temp_frame_num = feat_data.shape[0]
                total_num.assign_add(temp_frame_num)

                temp_feat = tf.reshape(feat_data, [-1, feature_dim])
                temp_feat2 = tf.square(temp_feat)

                temp_mean = tf.reduce_sum(temp_feat, axis=[0])
                temp_var = tf.reduce_sum(temp_feat2, axis=[0])

                initial_mean.assign_add(temp_mean)
                initial_var.assign_add(temp_var)
            # save the sums for per speaker in a dict
            initial_mean_dict[tar_speaker] = initial_mean
            initial_var_dict[tar_speaker] = initial_var
            total_num_dict[tar_speaker] = total_num
        return initial_mean_dict, initial_var_dict, total_num_dict

    def compute_cmvn_kaldiio(self, entries, speakers, kaldi_io_feats, feature_dim):
        """compute cmvn for filtered entries using kaldi-format data
        """
        start = time.time()
        for tar_speaker in set(speakers.values()):
            logging.info("processing %s" % tar_speaker)
            initial_mean = tf.Variable(tf.zeros([feature_dim], dtype=tf.float32))
            initial_var = tf.Variable(tf.zeros([feature_dim], dtype=tf.float32))
            total_num = tf.Variable(0, dtype=tf.int32)

            tq_entries = tqdm.tqdm(entries)
            for items in tq_entries:
                key, speaker = items
                if speaker != tar_speaker:
                    continue
                feat_data = kaldi_io_feats[key]
                feat_data = tf.convert_to_tensor(feat_data)
                temp_frame_num = feat_data.shape[0]
                total_num.assign_add(temp_frame_num)

                temp_feat = tf.reshape(feat_data, [-1, feature_dim])
                temp_feat2 = tf.square(temp_feat)

                temp_mean = tf.reduce_sum(temp_feat, axis=[0])
                temp_var = tf.reduce_sum(temp_feat2, axis=[0])

                initial_mean.assign_add(temp_mean)
                initial_var.assign_add(temp_var)

            # compute mean and var
            if total_num == 0:
                continue
            total_num = tf.cast(total_num, tf.float32)
            mean = initial_mean / total_num
            variance = initial_var / total_num - tf.square(mean)
            self.cmvn_dict[tar_speaker] = (list(mean.numpy()), list(variance.numpy()))

        logging.info("finished compute cmvn, which cost %.4f s" % (time.time() - start))

    def load_cmvn(self):
        """load mean and var
        """
        if not os.path.exists(self.cmvn_file):
            return
        cmvns = pandas.read_csv(self.cmvn_file, sep="\t", index_col="speaker")
        for speaker, cmvn in cmvns.iterrows():
            self.cmvn_dict[speaker] = (
                json.loads(cmvn["mean"]),
                json.loads(cmvn["var"]),
            )
        logging.info("Successfully load cmvn file {}".format(self.cmvn_file))

    def save_cmvn(self, variable_list):
        """save cmvn variables determined by variable_list to file

        Args:
            variable_list (list): e.g. ["speaker", "mean", "var"]
        """
        if self.cmvn_file is None:
            self.cmvn_file = "~/.athena/cmvn_file"
        cmvn_dir = os.path.dirname(self.cmvn_file)
        if not os.path.exists(cmvn_dir):
            os.mkdir(cmvn_dir)
        cmvns = []
        for speaker in self.cmvn_dict:
            temp = [speaker]
            temp.extend(self.cmvn_dict[speaker])
            cmvns.append(temp)
        df = pandas.DataFrame(data=cmvns, columns=variable_list)
        df.to_csv(self.cmvn_file, index=False, sep="\t")
        logging.info("Successfully save cmvn file {}".format(self.cmvn_file))

        
class WorldFeatureNormalizer(FeatureNormalizer):
    """World Feature Normalizer
    """

    def compute_world_cmvn(self, enable_load_from_disk, entries_person_wavs, sp_dim, fft_size, fs, speakers):
        """compuate cmvn of f0 and sp using pyworld
        """
        start = time.time()
        cmvns = []
        for speaker in speakers:
            coded_sps, f0s = [], []
            for audio_file in entries_person_wavs[speaker]:
                # World Vocoder parameterizes speech into three components:
                #     Pitch (fundamental frequency, F0) contour
                #     Harmonic spectral envelope(sp)
                #     Aperiodic spectral envelope (relative to the harmonic spectral envelope,ap)
                # Refer to the address：https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
                if enable_load_from_disk:
                    samples = np.load(audio_file)
                    f0, coded_sp, ap = samples["f0"], samples["coded_sp"], samples["ap"]
                else:
                    wav, _ = librosa.load(audio_file, sr=fs, mono=True, dtype=np.float64)
                    f0, timeaxis = pyworld.harvest(wav, fs)
                    # CheapTrick harmonic spectral envelope estimation algorithm.
                    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs, fft_size=fft_size)
                    # feature reduction
                    coded_sp = pyworld.code_spectral_envelope(sp, fs, sp_dim).T
                coded_sps.append(coded_sp)
                f0s.append(np.reshape(f0, [-1, 1]))
            # Calculate the mean and standard deviation of the World features
            coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
            coded_sps_mean = list(np.mean(coded_sps_concatenated, axis=1, keepdims=False))
            coded_sps_var = list(np.var(coded_sps_concatenated, axis=1, keepdims=False))
            log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
            log_f0s_mean = log_f0s_concatenated.mean()
            log_f0s_var = log_f0s_concatenated.var()
            cmvns.append((speaker, coded_sps_mean, coded_sps_var, log_f0s_mean, log_f0s_var))
            self.cmvn_dict[speaker] = (coded_sps_mean, coded_sps_var, \
                                       log_f0s_mean, log_f0s_var)

        logging.info("finished compute cmvn, which cost %.4f s" % (time.time() - start))

    def load_cmvn(self):
        """load codedsp_mean, codedsp_var, f0_mean, f0_var for vc dataset
        """
        if not os.path.exists(self.cmvn_file):
            return
        cmvns = pandas.read_csv(self.cmvn_file, sep="\t", index_col="speaker")
        for speaker, cmvn in cmvns.iterrows():
            self.cmvn_dict[speaker] = [
                json.loads(cmvn["codedsp_mean"]),
                json.loads(cmvn["codedsp_var"]),
            ]
            self.cmvn_dict[speaker].append(cmvn["f0_mean"])
            self.cmvn_dict[speaker].append(cmvn["f0_var"])
        logging.info("Successfully load cmvn file {}".format(self.cmvn_file))
