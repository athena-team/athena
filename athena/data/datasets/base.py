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
""" base dataset """

import math
import random
import os
from absl import logging
import tensorflow as tf
from athena.transform import AudioFeaturizer
from ..feature_normalizer import FeatureNormalizer
from ...utils.hparam import register_and_parse_hparams
from ...utils.data_queue import DataQueue


def data_loader(dataset_builder, batch_size=16, num_threads=1):
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
            capacity=4096,
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
        output_shapes=dataset_builder.sample_shape,
    )

    # Padding the features to its max length dimensions.
    dataset = dataset.padded_batch(
        batch_size=batch_size,
        padded_shapes=dataset_builder.sample_shape,
        drop_remainder=True,
    )

    # Prefetch to improve speed of input pipeline.
    dataset = dataset.prefetch(buffer_size=500)
    return dataset


class BaseDatasetBuilder:
    """base dataset builder
    """
    default_config = {}

    def __init__(self, config=None):
        # hparams
        self.hparams = register_and_parse_hparams(
            self.default_config, config, cls=self.__class__)
        logging.info("hparams: {}".format(self.hparams))
        self.entries = []

    def reload_config(self, config):
        """ reload the config """
        if config is not None:
            self.hparams.override_from_dict(config)

    def preprocess_data(self, file_path):
        """ loading data """
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.entries)

    @property
    def sample_type(self):
        """example types
        """
        raise NotImplementedError

    @property
    def sample_shape(self):
        """examples shapes
        """
        raise NotImplementedError

    @property
    def sample_signature(self):
        """examples signature
        """
        raise NotImplementedError

    def as_dataset(self, batch_size=16, num_threads=1):
        """return tf.data.Dataset object
        """
        return data_loader(self, batch_size, num_threads)

    def shard(self, num_shards, index):
        """creates a Dataset that includes only 1/num_shards of this dataset
        """
        if index >= num_shards:
            raise ValueError("the index should smaller the num_shards")
        logging.info("Creates the sub-dataset which is the %d part of %d" % (index, num_shards))
        original_entries = self.entries
        self.entries = []
        total_samples = (len(original_entries) // num_shards) * num_shards
        for i in range(total_samples):
            if i % num_shards == index:
                self.entries.append(original_entries[i])
        return self

    def batch_wise_shuffle(self, batch_size=64):
        """Batch-wise shuffling of the data entries.

        Each data entry is in the format of (audio_file, file_size, transcript).
        If epoch_index is 0 and sortagrad is true, we don't perform shuffling and
        return entries in sorted file_size order. Otherwise, do batch_wise shuffling.

        Args:
            batch_size (int, optional):  an integer for the batch size. Defaults to 64.
        """
        if len(self.entries) == 0:
            return self
        logging.info("perform batch_wise_shuffle with batch_size %d" % batch_size)
        max_buckets = int(math.floor(len(self.entries) / batch_size))
        total_buckets = list(range(max_buckets))
        random.shuffle(total_buckets)
        shuffled_entries = []
        for i in total_buckets:
            shuffled_entries.extend(self.entries[i * batch_size : (i + 1) * batch_size])
        shuffled_entries.extend(self.entries[max_buckets * batch_size :])
        self.entries = shuffled_entries
        return self

    def compute_cmvn_if_necessary(self, is_necessary=True):
        """ compute cmvn file
        """
        return self


class SpeechBaseDatasetBuilder(BaseDatasetBuilder):
    """ speech base dataset """
    default_config = {
        "audio_config": {"type": "Fbank"},
        "num_cmvn_workers": 1,
        "cmvn_file": None,
        "data_csv": None
    }

    def __init__(self, config=None):
        super().__init__(config=config)
        self.speakers = []
        self.audio_featurizer = AudioFeaturizer(self.hparams.audio_config)
        self.feature_normalizer = FeatureNormalizer(self.hparams.cmvn_file)

    @property
    def num_class(self):
        """ return the number of classes """
        raise NotImplementedError

    def compute_cmvn_if_necessary(self, is_necessary=True):
        """vitural interface
        """
        if not is_necessary:
            return self
        if os.path.exists(self.hparams.cmvn_file):
            return self
        feature_dim = self.audio_featurizer.dim * self.audio_featurizer.num_channels
        with tf.device("/cpu:0"):
            self.feature_normalizer.compute_cmvn(
                self.entries, self.speakers, self.audio_featurizer, feature_dim,
                self.hparams.num_cmvn_workers
            )
        self.feature_normalizer.save_cmvn(["speaker", "mean", "var"])
        return self
