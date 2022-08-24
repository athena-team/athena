# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Jianwei Sun
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
import tensorflow as tf
from athena.data.datasets.asr.speech_recognition import SpeechRecognitionDatasetBuilder
from athena.utils.num_elements_batch_sampler import NumElementsBatchSampler
from athena.utils.data_queue import DataQueue


def data_loader(dataset_builder, batch_size=1, num_threads=1):
    """data loader
    """
    num_samples = len(dataset_builder)
    if num_samples == 0:
        raise ValueError("num samples is empty")

    if num_threads == 1:
        def _gen_data():
            """multi thread loader
            """
            for i in range(num_samples):
                yield dataset_builder[i]
    else:
        # multi-thread
        logging.info("loading data using %d threads" % num_threads)
        data_queue = DataQueue(
            lambda i: dataset_builder[i],
            capacity=128,
            num_threads=num_threads,
            max_index=num_samples
        )

        def _gen_data():
            """multi thread loader
            """
            for _ in range(num_samples):
                yield data_queue.get()

    # make dataset using from_generator
    dataset = tf.compat.v2.data.Dataset.from_generator(
        _gen_data,
        output_types=dataset_builder.sample_type,
        output_shapes=dataset_builder.sample_shape_batch_bins,
    )

    # Prefetch to improve speed of input pipeline.
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def video_scp_loader(scp_dir):
    '''
    load video list from scp file
    return a dic
    '''
    file = open(scp_dir,'r')
    dic = {}
    for video_line in file:
        key,value = video_line.strip('\n').split('\t')
        dic[key] = value
    file.close()
    return dic
'''
def image_normalizer1(image):
    image = image / tf.reduce_max(image)
    return image
'''

def image_normalizer(image):
    """
    Image Normalization
    http://dev.ipol.im/~nmonzon/Normalization.pdf
    """
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))
    return image

class AudioVedioRecognitionDatasetBatchBinsBuilder(SpeechRecognitionDatasetBuilder):
    """SpeechRecognitionDatasetBatchBinsBuilder
    """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "text_config": {"type":"vocab", "model":"athena/utils/vocabs/ch-en.vocab"},
        "num_cmvn_workers": 1,
        "cmvn_file": None,
        "remove_unk": True,
        "input_length_range": [0, 50000],
        "output_length_range": [1, 10000],
        "speed_permutation": [1.0],
        "spectral_augmentation": None,
        "data_csv": None,
        "data_scp": None,
        "words": None,
        "apply_cmvn": True,
        "global_cmvn": True,
        "offline": False,
        "ignore_unk": False,
        "mini_batch_size": 32,
        "batch_bins": 4200000,
        "image_shape": [100, 150, 1],
        "video_skip": 4,
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    def preprocess_data(self, file_path):
        super().preprocess_data(file_path)
        shape_files = ["speech_shape", "text_shape"]
        utt2shapes = [self.speech_shape_dict, self.text_shape_dict]

        self.batchs_list = NumElementsBatchSampler(
            batch_bins=self.hparams.batch_bins, shape_files=shape_files, utt2shapes=utt2shapes, min_batch_size=self.hparams.mini_batch_size)
        self.batchs_list = list(self.batchs_list)
        self.video_list = video_scp_loader(self.hparams.data_scp)
        self.high, self.wide, self.channel = self.hparams.image_shape
        return self

    def __getitem__(self, index):
        feats = []
        labels = []
        videos = []
        feat_length = []
        label_length = []
        utts = []
        for audio_data, speed, transcripts, speaker in self.batchs_list[index]:
            feat = self.audio_featurizer(audio_data, speed=speed)
            key = audio_data.split('/')[-1].split('.')[0]
            video = np.load(self.video_list[key])
            video = tf.convert_to_tensor(video)
            video = image_normalizer(video)
            if self.channel==1:
                video = tf.expand_dims(tf.convert_to_tensor(video), -1)

            if self.hparams.apply_cmvn:
                feat = self.feature_normalizer(feat, speaker)
            if self.hparams.spectral_augmentation is not None:
                feat = tf.squeeze(feat).numpy()
                #print(audio_data)#debug
                feat = self.spectral_augmentation(feat)
                #print("augmentation ok")#debug
                feat = tf.expand_dims(tf.convert_to_tensor(feat), -1)
            #video = video[1:math.ceil(feat.shape[0]/self.hparams.video_skip)+1,:,:]
            videos.append(video)

            label = self.text_featurizer.encode(transcripts)

            feat_length.append(feat.shape[0])
            label_length.append(len(label))
            feats.append(feat)
            labels.append(label)
            utts.append(audio_data)

        n_batch = len(feats)
        max_len = max(x.shape[0] for x in feats)
        feats_pad = np.zeros(
            [n_batch, max_len, feats[0].shape[1], feats[0].shape[2]], dtype=np.float)

        #video
        #n_batch = len(videos)
        max_len = max(x.shape[0] for x in videos)
        videos_pad = np.zeros(
            [n_batch, max_len, videos[0].shape[1], videos[0].shape[2], self.channel], dtype=np.float)


        max_len = max(len(x)for x in labels)
        labels_pad = np.zeros(
            [n_batch, max_len], dtype=np.int)

        for i in range(n_batch):
            feats_pad[i, : feats[i].shape[0]] = feats[i]
            labels_pad[i, : len(labels[i])] = labels[i]
            videos_pad[i,: videos[i].shape[0],:] = videos[i]

        return {
            "input": feats_pad,
            "video": videos_pad,
            "input_length": np.array(feat_length),
            "output_length": np.array(label_length),
            "output": labels_pad,
            "utt_id": np.array(utts),
        }

    def __len__(self):
        return len(self.batchs_list)

    @property
    def sample_shape_batch_bins(self):
        """:obj:`@property`

        Returns:
            dict: sample_shape of the dataset::

            {
                "input": tf.TensorShape([None, None, dim, nc]),
                "video":tf.TensorShape([None, None, high, wide]),
                "input_length": tf.TensorShape([None]),
                "output_length": tf.TensorShape([None]),
                "output": tf.TensorShape([None, None]),
                "utt_id": tf.TensorShape([None]),
            }
        """
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return {
            "input": tf.TensorShape([None, None, dim, nc]),
            "video": tf.TensorShape([None, None, self.high, self.wide, self.channel]),
            "input_length": tf.TensorShape([None]),
            "output_length": tf.TensorShape([None]),
            "output": tf.TensorShape([None, None]),
            "utt_id": tf.TensorShape([None]),
        }

    def as_dataset(self, batch_size=16, num_threads=1):
        """return tf.data.Dataset object
        """
        return data_loader(self, batch_size, num_threads)

    def shard(self, num_shards, index):
        """creates a Dataset that includes only 1/num_shards of this dataset
        """
        return self

    def batch_wise_shuffle(self, batch_size=1, epoch=-1, seed=917):
        """Batch-wise shuffling of the data entries.

        Args:
            batch_size (int, optional):  an integer for the batch size. Defaults to 1
            in batch_bins mode .
        """
        if len(self.entries) == 0 or epoch < 0:
            return self

        logging.info(
            "perform batch_wise_shuffle with batch_size %d" % batch_size)
        np.random.RandomState(epoch + seed).shuffle(self.batchs_list)
        return self
    @property
    def sample_shape(self):
        """:obj:`@property`

        Returns:
            dict: sample_shape of the dataset::

            {
                "input": tf.TensorShape([None, dim, nc]),
                "video": tf.TensorShape([None, None, None]),
                "input_length": tf.TensorShape([]),
                "output_length": tf.TensorShape([]),
                "output": tf.TensorShape([None]),
                "utt_id": tf.TensorShape([]),
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
            "utt_id": tf.TensorShape([]),
        }

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
                "utt_id": tf.string,
            }
        """
        return {
            "input": tf.float32,
            "video": tf.float32,
            "input_length": tf.int32,
            "output_length": tf.int32,
            "output": tf.int32,
            "utt_id": tf.string,
        }

    @property
    def sample_signature(self):
        """:obj:`@property`

        Returns:
            dict: sample_signature of the dataset::

            {
                "input": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "video": tf.TensorSpec(shape=(None, None, None, None), dtype=tf.float32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "utt_id": tf.TensorSpec(shape=(None), dtype=tf.string),
            }
        """
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return (
            {
                "input": tf.TensorSpec(shape=(None, None, dim, nc), dtype=tf.float32),
                "video": tf.TensorSpec(shape=(None, None, self.high, self.wide,self.channel), dtype=tf.float32),
                "input_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output_length": tf.TensorSpec(shape=(None), dtype=tf.int32),
                "output": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                "utt_id": tf.TensorSpec(shape=(None), dtype=tf.string),
            },
        )
