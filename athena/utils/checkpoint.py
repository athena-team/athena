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
# Only support eager mode
# pylint: disable=invalid-name
r""" checkpoint manager """
import os
import shutil

import tensorflow as tf
from absl import logging
from ..utils.misc import get_dict_from_scp, read_csv_dict, write_csv
import numpy as np


class Checkpoint(tf.train.Checkpoint):
    """A wrapper for Tensorflow checkpoint

    Args:
        checkpoint_directory: the directory for checkpoint
        summary_directory: the directory for summary used in Tensorboard
        __init__: provide the optimizer and model
        __call__: save the model

    Example:

        >>> transformer = SpeechTransformer(target_vocab_size=dataset_builder.target_dim)
        >>> optimizer = tf.keras.optimizers.Adam()
        >>> ckpt = Checkpoint(checkpoint_directory='./train', summary_directory='./event',
        >>>        transformer=transformer, optimizer=optimizer)
        >>> solver = BaseSolver(transformer)
        >>> for epoch in dataset:
        >>>    ckpt()
    """

    def __init__(self, checkpoint_directory=None, use_dev_loss=True, model=None, **kwargs):
        super().__init__(**kwargs, model=model)
        if checkpoint_directory is None:
            checkpoint_directory = os.path.join(os.path.expanduser("~"), ".athena")
        self.checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        self.checkpoint_directory = checkpoint_directory

        self.best_loss_old_path = os.path.join(self.checkpoint_directory, "best_loss")
        self.best_loss_train_path = os.path.join(self.checkpoint_directory, "best_loss_train")
        self.best_loss_dev_path = os.path.join(self.checkpoint_directory, "best_loss_dev")
        self.n_best_old_path = os.path.join(self.checkpoint_directory, "n_best")
        self.n_best_train_path = os.path.join(self.checkpoint_directory, "n_best_train")
        self.n_best_dev_path = os.path.join(self.checkpoint_directory, "n_best_dev")

        self.n_best_path = self.n_best_dev_path if use_dev_loss else self.n_best_train_path
        self.best_loss_path = self.best_loss_dev_path if use_dev_loss else self.best_loss_train_path

        self.metric_name = model.metric.name

        logging.info("trying to restore from : %s" % checkpoint_directory)
        # load from checkpoint if previous models exist in checkpoint dir
        self.restore(tf.train.latest_checkpoint(checkpoint_directory))
        # compatible with previous versions
        self._file_compatible(use_dev_loss)

        # restore best_loss and n_best
        loss_train_local = list(get_dict_from_scp(self.best_loss_train_path, lambda x: float(x[0])).values()) \
            if os.path.exists(self.best_loss_train_path) else []
        self.best_loss_train = loss_train_local[-1] if len(loss_train_local) > 0 else np.inf
        loss_dev_local = list(get_dict_from_scp(self.best_loss_dev_path, lambda x: float(x[0])).values()) \
            if os.path.exists(self.best_loss_dev_path) else []
        self.best_loss_dev = loss_dev_local[-1] if len(loss_dev_local) > 0 else np.inf

        # self.n_best_model_train = get_dict_from_scp(self.n_best_train_path, lambda x: float(x[0])) \
        #     if os.path.exists(self.n_best_train_path) else {}
        # self.n_best_model_dev = get_dict_from_scp(self.n_best_dev_path, lambda x: float(x[0])) \
        #     if os.path.exists(self.n_best_dev_path) else {}
        self.n_best_model_train = read_csv_dict(self.n_best_train_path) if os.path.exists(self.n_best_train_path) else []
        self.n_best_model_dev = read_csv_dict(self.n_best_dev_path) if os.path.exists(self.n_best_dev_path) else []

    def _file_compatible(self, use_dev_loss):
        """Convert  n_best file to CSV file

        Add "index" and "Accuracy" for no csv n_best file.
        """
        # not exists
        if not os.path.exists(self.n_best_path):
            return
        # or is new
        with open(self.n_best_path, "r")as fin:
            lines = fin.readlines()
            # self.n_best_path is csv files, do not need to be converted
            if "index" in lines[0]:
                return
        # just return

        # when (self.n_best_path + ".new") is generated and it is the new one, just use it.
        if os.path.exists(self.n_best_path + ".new") and \
            os.path.getmtime(self.n_best_path + ".new") > os.path.getmtime(self.n_best_path):
            self.n_best_path = self.n_best_path + ".new"
            self.n_best_train_path = self.n_best_train_path + ".new"
            self.n_best_dev_path = self.n_best_dev_path + ".new"
            return

        # generate (self.n_best_path + ".new") use self.n_best_path
        for path in [self.n_best_train_path, self.n_best_dev_path]:
            if os.path.exists(path):
                n_best_path_old = path
                n_best_path_new = path + ".new"
                nbest = get_dict_from_scp(n_best_path_old, lambda x: float(x[0]))

                res = [{"index": index, "Accuracy": nbest[index]} for index in nbest]
                write_csv(n_best_path_new, res)

        self.n_best_path = self.n_best_path + ".new"
        self.n_best_train_path = self.n_best_train_path + ".new"
        self.n_best_dev_path = self.n_best_dev_path + ".new"
        return

    def _compare_and_save_best(self, loss, metrics, save_path, training=False):
        """compare and save the best model with best_loss and N best metrics"""         
        checkpoint = save_path.split('/')[-1]
        loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
        if training:
            if loss is not None and loss < self.best_loss_train:
                self.best_loss_train = loss
                with open(self.best_loss_train_path, 'w') as wf:
                    wf.write('%s\t%s\n' % (checkpoint, float(self.best_loss_train)))
        else:
            if loss is not None and loss < self.best_loss_dev:
                self.best_loss_dev = loss
                with open(self.best_loss_dev_path, 'w') as wf:
                    wf.write('%s\t%s\n' % (checkpoint, float(self.best_loss_dev)))

        if metrics is None or len(metrics) == 0 or self.metric_name is None:
            return
        if training:
            n_best_model_local = self.n_best_model_train
            n_best_path_local = self.n_best_train_path
        else:
            n_best_model_local = self.n_best_model_dev
            n_best_path_local = self.n_best_dev_path

        n_best_model_local.append({"index": checkpoint, "loss": float(loss),
                                   **{key: float(values) for key, values in metrics.items()}})
        write_csv(n_best_path_local, n_best_model_local)

    def compute_nbest_avg(self, model_avg_num, sort_by=None, sort_by_time=False, reverse=True):
        """Restore n-best avg checkpoint,

        if 'sort_by_time' is False, the n-best order is sorted by 'sort_by';
        If 'sort_by_time' is True, select the newest few models;
        If 'reverse' is True, select the largest models in the sorted order;
        """
        avg_file = self.n_best_path
        if not os.path.exists(avg_file):
            self.restore_from_best()
            return
        ckpt_metrics_dict = read_csv_dict(avg_file)
        if sort_by is None:
            sort_by = list(ckpt_metrics_dict[0].keys())[-1]  # choose the last column in n_best_dev file
        if not sort_by_time:
            ckpt_sorted_list = sorted(ckpt_metrics_dict, key=lambda item: float(item[sort_by]), reverse=reverse)
        else:
            ckpt_sorted_list = sorted(ckpt_metrics_dict, key=lambda item: int((item["index"]).split("-")[1]), reverse=reverse)
        ckpt_list = [item["index"] for item in ckpt_sorted_list][0: model_avg_num]
        logging.info('n_best_metrics_checkpoint: %s' % ckpt_list)
        ckpt_v_list = []
        # restore v from ckpts
        for key in ckpt_list:
            ckpt_path = os.path.join(self.checkpoint_directory, key)
            self.restore(ckpt_path)  # current variables will be updated
            var_list = []
            for i in self.model.trainable_variables:
                v = tf.constant(i.value())
                var_list.append(v)
            ckpt_v_list.append(var_list)
        # compute average, and assign to current variables
        for i in range(len(self.model.trainable_variables)):
            v = [tf.expand_dims(ckpt_v_list[j][i], [0]) for j in range(len(ckpt_v_list))]
            v = tf.reduce_mean(tf.concat(v, axis=0), axis=0)
            self.model.trainable_variables[i].assign(v)

    def __call__(self, loss=None, metrics=None, training=False):
        logging.info("saving model in :%s" % self.checkpoint_prefix)
        save_path = self.save(file_prefix=self.checkpoint_prefix)
        self._compare_and_save_best(loss, metrics, save_path, training)

    def restore_from_best(self):
        """restore from the best model"""
        self.restore(
            tf.train.latest_checkpoint(
                self.checkpoint_directory,
                latest_filename=self.best_loss_path.split("/")[-1]
            )
        )
