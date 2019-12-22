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
import tqdm
import time
import pandas
from absl import logging
import tensorflow as tf


class FeatureNormalizer:
    """ Feature Normalizer """

    def __init__(self, cmvn_file=None):
        super().__init__()
        self.cmvn_file = cmvn_file
        self.cmvn_dict = {}
        self.speakers = []
        if cmvn_file is not None:
            self.load_cmvn()

    def __call__(self, feat_date, speaker):
        return self.apply_cmvn(feat_date, speaker)

    def apply_cmvn(self, feat_data, speaker):
        """ TODO: docstring"""
        if speaker not in self.cmvn_dict:
            return feat_data
        mean = self.cmvn_dict[speaker][0]
        var = self.cmvn_dict[speaker][1]
        shape = feat_data.get_shape().as_list()[1:]
        mean = tf.reshape(tf.convert_to_tensor(mean, dtype=tf.float32), shape)
        var = tf.reshape(tf.convert_to_tensor(var, dtype=tf.float32), shape)
        feat_data = (feat_data - mean) / tf.sqrt(var)
        return feat_data

    def compute_cmvn(self, entries, speakers, featurizer, feature_dim):
        """ Compute cmvn for filtered entries """
        start = time.time()
        for tar_speaker in speakers:
            logging.info("processing %s" % tar_speaker)
            initial_mean = tf.Variable(tf.zeros([feature_dim], dtype=tf.float32))
            initial_var = tf.Variable(tf.zeros([feature_dim], dtype=tf.float32))
            total_num = tf.Variable(0, dtype=tf.int32)

            tq_entries = tqdm.tqdm(entries)
            for items in tq_entries:
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

            # compute mean and var
            if total_num == 0:
                continue
            total_num = tf.cast(total_num, tf.float32)
            mean = initial_mean / total_num
            variance = initial_var / total_num - tf.square(mean)
            self.cmvn_dict[tar_speaker] = (list(mean.numpy()), list(variance.numpy()))

        logging.info("finished compute cmvn, which cost %.4f s" % (time.time() - start))

    def load_cmvn(self):
        """ TODO: docstring """
        if not os.path.exists(self.cmvn_file):
            return
        cmvns = pandas.read_csv(self.cmvn_file, sep="\t", index_col="speaker")
        for speaker, cmvn in cmvns.iterrows():
            self.cmvn_dict[speaker] = (
                json.loads(cmvn["mean"]),
                json.loads(cmvn["var"]),
            )
        logging.info("Successfully load cmvn file {}".format(self.cmvn_file))

    def save_cmvn(self):
        """ TODO: docstring """
        if self.cmvn_file is None:
            self.cmvn_file = "~/.athena/cmvn_file"
        cmvn_dir = os.path.dirname(self.cmvn_file)
        if not os.path.exists(cmvn_dir):
            os.mkdir(cmvn_dir)
        cmvns = []
        for speaker in self.cmvn_dict:
            cmvns.append(
                (speaker, self.cmvn_dict[speaker][0], self.cmvn_dict[speaker][1])
            )
        df = pandas.DataFrame(data=cmvns, columns=["speaker", "mean", "var"])
        df.to_csv(self.cmvn_file, index=False, sep="\t")
        logging.info("Successfully save cmvn file {}".format(self.cmvn_file))
