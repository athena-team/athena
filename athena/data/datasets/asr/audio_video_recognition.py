# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Shuaijiang Zhao; Jianwei Sun
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
import sys
import numpy as np
import kaldiio
from absl import logging
import tensorflow as tf
from athena.data.text_featurizer import TextFeaturizer
from athena.data.datasets.base import SpeechBaseDatasetBuilder
from athena.data.datasets.preprocess import SpecAugment
import os.path

class AudioVedioRecognitionDatasetBuilder(SpeechBaseDatasetBuilder):
    """SpeechRecognitionDatasetBuilder
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "text_config": {"type": "vocab", "model": "athena/utils/vocabs/ch-en.vocab"},
        "num_cmvn_workers": 1,
        "cmvn_file": None,
        "remove_unk": True,
        "input_length_range": [0, 50000],
        "output_length_range": [1, 10000],
        "speed_permutation": [1.0],
        "spectral_augmentation": None,
        "data_csv": None,
        "data_scp": None,
        "data_scps_dir": None,
        "words": None,
        "apply_cmvn": True,
        "global_cmvn": True,
        "offline": False,
        "ignore_unk": False,
        "image_shape": [100, 150, 1],
        "video_skip": 4,
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        if self.hparams.data_csv == None:
            pass
            logging.warning("created a fake dataset")
        self.text_featurizer = TextFeaturizer(self.hparams.text_config)
        if self.hparams.spectral_augmentation is not None:
            self.spectral_augmentation = SpecAugment(
                self.hparams.spectral_augmentation)
        if self.hparams.data_csv is not None:
            self.preprocess_data(self.hparams.data_csv)

    def video_scp_loader(self, scp_dir):
        '''
        load video list from scp file
        return a dic
        '''
        file = open(scp_dir, 'r')
        dic = {}
        for video_line in file:
            key, value = video_line.strip('\n').split('\t')
            dic[key] = value
        file.close()
        return dic

    def image_normalizer(self, image):
        image = image / tf.reduce_max(image)
        return image

    def preprocess_data(self, file_path):
        """generate a list of tuples (wav_filename, wav_length_ms, transcript, speaker).
        """
        logging.info("Loading data from {}".format(file_path))
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()
        headers = lines[0]
        lines = lines[1:]
        lines = [line.split("\t") for line in lines]
        self.entries = [tuple(line) for line in lines]

        # handling speakers
        self.speakers = []
        if self.hparams.global_cmvn or "speaker" not in headers.split("\t"):
            entries = self.entries
            self.entries = []
            for wav_filename, wav_len, transcripts, _ in entries:
                self.entries.append(
                    tuple([wav_filename, wav_len, transcripts, "global"])
                )
            self.speakers.append("global")
        else:
            for _, _, _, speaker in self.entries:
                if speaker not in self.speakers:
                    self.speakers.append(speaker)

        self.speech_shape_dict = {}
        self.text_shape_dict = {}
        # handling speed
        entries = self.entries
        self.entries = []
        if len(self.hparams.speed_permutation) > 1:
            logging.info("perform speed permutation")
        for speed in self.hparams.speed_permutation:
            for wav_filename, wav_len, transcripts, speaker in entries:
                self.entries.append(
                    tuple([wav_filename,
                    float(wav_len) /
                    float(speed), transcripts, speed, speaker
                    ]))

        self.entries.sort(key=lambda item: float(item[1]))
        # handling special case for text_featurizer
        if self.text_featurizer.model_type == "text":
            _, _, all_transcripts, _, _ = zip(*self.entries)
            self.text_featurizer.load_model(all_transcripts)

        # apply some filter
        unk = self.text_featurizer.unk_index
        if self.hparams.remove_unk and unk != -1 and self.hparams.ignore_unk == False:
            self.entries = list(filter(lambda x: unk not in
                                self.text_featurizer.encode(x[2]), self.entries))
        self.entries = list(filter(lambda x: int(x[1]) in
                            range(self.hparams.input_length_range[0],
                            self.hparams.input_length_range[1]), self.entries))
        self.entries = list(filter(lambda x: len(x[2]) in
                            range(self.hparams.output_length_range[0],
                            self.hparams.output_length_range[1]), self.entries))
        
        for wav_filename, wav_len, transcripts, speed, speaker in self.entries:
            self.speech_shape_dict[tuple([wav_filename, speed, transcripts, speaker])] = [
                wav_len * 16 * 2]  # 16 is wav freq, ex 16k or 8k
            self.text_shape_dict[tuple([wav_filename, speed, transcripts, speaker])] = [
                len(transcripts) * 2]

        self.video_list = self.video_scp_loader(self.hparams.data_scp)
        self.high, self.wide, self.channel = self.hparams.image_shape

        return self

    def storage_features_offline(self):
        feat_dict = {}
        label_dict = {}
        feat_shape_file = open(self.hparams.data_scps_dir + "/" + "feat_shape", 'w')
        label_shape_file = open(self.hparams.data_scps_dir + "/" + "label_shape", 'w')
        for key, _, transcripts, speed, speaker in self.entries:
            feat = self.audio_featurizer(key, speed=speed)
            if self.hparams.apply_cmvn:
                if os.path.exists(self.hparams.cmvn_file):
                    feat = self.feature_normalizer(feat, speaker)
                else:
                    logging.warning("{} does not exist".format(
                        self.hparams.cmvn_file))
            if self.hparams.spectral_augmentation is not None:
                feat = tf.squeeze(feat).numpy()
                feat = self.spectral_augmentation(feat)
                feat = tf.expand_dims(tf.convert_to_tensor(feat), -1)
            feat = np.reshape(
                feat, (-1, self.hparams.audio_config["filterbank_channel_count"]))
            label = self.text_featurizer.encode(transcripts)
            if str(speed) == "1.0":
                key = os.path.basename(key).replace(".wav", "")
            else:
                key = os.path.basename(key).replace(".wav", "_"+str(speed))
            feat_dict[key] = feat
            label_dict[key] = np.array(label, dtype='int32')
            feat_shape_file.write(key + '\t' + speaker + '\t' + str(feat.shape[0] * feat.shape[1] * 4) + '\n')
            label_shape_file.write(key + '\t' + speaker + '\t' + str(len(label) * 4) + '\n')


        kaldiio.save_ark(os.path.join(self.hparams.data_scps_dir, "feats.ark"),
                         feat_dict, scp=os.path.join(self.hparams.data_scps_dir, "feats.scp"))
        kaldiio.save_ark(os.path.join(self.hparams.data_scps_dir, "labels.ark"),
                         label_dict, scp=os.path.join(self.hparams.data_scps_dir, "labels.scp"))
        return self

    def __getitem__(self, index):
        """get a sample

        Args:
            index (int): index of the entries

        Returns:
            dict: sample::

            {
                "input": feat,
                "input_length": feat_length,
                "output_length": label_length,
                "output": label,
            }
        """
        audio_data, _, transcripts, speed, speaker = self.entries[index]
        feat = self.audio_featurizer(audio_data, speed=speed)
        key = audio_data.split('/')[-1].split('.')[0]
        video = np.load(self.video_list[key])
        video = tf.convert_to_tensor(video)
        video = self.image_normalizer(video)
        if self.channel == 1:
            video = tf.expand_dims(tf.convert_to_tensor(video), -1)
        if self.hparams.apply_cmvn:
            feat = self.feature_normalizer(feat, speaker)
        if self.hparams.spectral_augmentation is not None:
            feat = tf.squeeze(feat).numpy()
            feat = self.spectral_augmentation(feat)
            feat = tf.expand_dims(tf.convert_to_tensor(feat), -1)
        feat_length = feat.shape[0]

        label = self.text_featurizer.encode(transcripts)
        label_length = len(label)
        # audio_name = tf.Variable(os.path.basename(audio_data), dtype=tf.string)
        utt = os.path.basename(audio_data)
        return {
            "input": feat,
            "video": video,
            "input_length": feat_length,
            "output_length": label_length,
            "output": label,
            "utt": utt
        }

    @property
    def num_class(self):
        """ return the max_index of the vocabulary + 1 """
        return len(self.text_featurizer)

    @property
    def sample_type(self):
        """:obj:`@property`

        Returns:
            dict: sample_type of the dataset::

            {
                "input": tf.float32,
                "input_length": tf.int32,
                "output_length": tf.int32,
                "output": tf.int32,
            }
        """
        return {
            "input": tf.float32,
            "video": tf.float32,
            "input_length": tf.int32,
            "output_length": tf.int32,
            "output": tf.int32,
            "utt": tf.string,
        }

    @property
    def sample_shape(self):
        """:obj:`@property`

        Returns:
            dict: sample_shape of the dataset::

            {
                "input": tf.TensorShape([None, dim, nc]),
                "input_length": tf.TensorShape([]),
                "output_length": tf.TensorShape([]),
                "output": tf.TensorShape([None]),
            }
        """
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return {
            "input": tf.TensorShape([None, dim, nc]),
            "video": tf.TensorShape([None, self.high, self.wide, self.channel]),
            "input_length": tf.TensorShape([]),
            "output_length": tf.TensorShape([]),
            "output": tf.TensorShape([None]),
            "utt": tf.TensorShape([]),
        }

    @property
    def sample_signature(self):
        """:obj:`@property`

        Returns:
            dict: sample_signature of the dataset::

            {
                "input": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
            }
        """
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return (
            {
                "input": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "video": tf.TensorSpec(shape=(None, None, self.high, self.wide, self.channel), dtype=tf.float32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "utt": tf.TensorSpec(shape=(None), dtype=tf.string),
            },
        )

