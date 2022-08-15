# coding=utf-8
# Copyright (C) ATHENA AUTHORS; Xiangang Li; Yanguang Xu; Xuezheng Deng; Jianwei Sun
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
            local_rank = 0
            hvd_size = 1
            if os.getenv('HOROVOD_TRAIN_MODE', 'normal') == 'fast':
                try:
                    import horovod.tensorflow as hvd
                    local_rank = hvd.rank()
                    hvd_size = hvd.size()
                except ImportError:
                    print("There is some problem with your horovod installation. But it wouldn't affect single-gpu training")
            for i in range(local_rank, num_samples, hvd_size):
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


class SpeechRecognitionDatasetBatchBinsBuilder(SpeechRecognitionDatasetBuilder):
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
        "words": None,
        "apply_cmvn": True,
        "global_cmvn": True,
        "offline": False,
        "ignore_unk": False,
        "mini_batch_size": 32,
        "batch_bins": 4200000,
        "train_mode": 'normal_horovod',
        "rank_size": 1,
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
        if self.hparams.rank_size > 1 and self.hparams.train_mode == 'fast_horovod':
           os.environ['HOROVOD_TRAIN_MODE'] = 'fast'

    def preprocess_data(self, file_path):
        super().preprocess_data(file_path)
        shape_files = ["speech_shape", "text_shape"]
        utt2shapes = [self.speech_shape_dict, self.text_shape_dict]

        self.batchs_list = NumElementsBatchSampler(
            batch_bins=self.hparams.batch_bins, shape_files=shape_files, utt2shapes=utt2shapes, min_batch_size=self.hparams.mini_batch_size)
        self.batchs_list = list(self.batchs_list)
        return self

    def __getitem__(self, index):
        feats = []
        labels = []
        feat_length = []
        label_length = []
        utts = []
        for audio_data, speed, transcripts, speaker in self.batchs_list[index]:
            feat = self.audio_featurizer(audio_data, speed=speed) 
            if self.hparams.apply_cmvn:
                feat = self.feature_normalizer(feat, speaker)
            if self.hparams.spectral_augmentation is not None:
                feat = tf.squeeze(feat).numpy()
                try:
                    feat = self.spectral_augmentation(feat)
                except:
                    logging.error("badcase:")
                    logging.error(audio_data)
                feat = tf.expand_dims(tf.convert_to_tensor(feat), -1)
            label = self.text_featurizer.encode(transcripts)

            feat_length.append(feat.shape[0])
            label_length.append(len(label))
            feats.append(feat)
            labels.append(label)
            utts.append(os.path.basename(audio_data))

        n_batch = len(feats)
        max_len = max(x.shape[0] for x in feats)
        feats_pad = np.zeros(
            [n_batch, max_len, feats[0].shape[1], feats[0].shape[2]], dtype=np.float)

        max_len = max(len(x)for x in labels)
        labels_pad = np.zeros(
            [n_batch, max_len], dtype=np.int)

        for i in range(n_batch):
            feats_pad[i, : feats[i].shape[0]] = feats[i]
            labels_pad[i, : len(labels[i])] = labels[i]

        return {
            "input": feats_pad,
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
                "input_length": tf.TensorShape([None]),
                "output_length": tf.TensorShape([None]),
                "output": tf.TensorShape([None, None]),
            }
        """
        dim = self.audio_featurizer.dim
        nc = self.audio_featurizer.num_channels
        return {
            "input": tf.TensorShape([None, None, dim, nc]),
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
