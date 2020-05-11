# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li
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
# Only support tensorflow 2.0
# pylint: disable=invalid-name, no-member
r""" a sample implementation of LAS for HKUST """
import warnings
import sys
import json
import tensorflow as tf
import horovod.tensorflow as hvd
from absl import logging
from athena import BaseSolver
from athena.main import parse_config, train


class HorovodSolver(BaseSolver):
    """ A multi-processer solver based on Horovod """

    @staticmethod
    def initialize_devices(visible_gpu_idx=None):
        """ initialize hvd devices, should be called firstly """
        if visible_gpu_idx is not None:
            warnings.warn("we can not set the visible gpu idx like this")
        hvd.init()
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    def train_step(self, samples):
        """ train the model 1 step """
        with tf.GradientTape() as tape:
            logits = self.model(samples, training=True)
            loss, metrics = self.model.get_loss(logits, samples, training=True)
        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = self.clip_by_norm(grads, self.hparams.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, metrics

    def train(self, dataset, total_batches=-1):
        """ Update the model in 1 epoch """
        train_step = self.train_step
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            train_step = tf.function(train_step, input_signature=self.sample_signature)
        for batch, samples in enumerate(dataset.take(total_batches)):
            # train 1 step
            samples = self.model.prepare_samples(samples)
            loss, metrics = train_step(samples)
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            #
            # Note: broadcast should be done after the first gradient step to ensure optimizer
            # initialization.
            if batch == 0:
                hvd.broadcast_variables(self.model.trainable_variables, root_rank=0)
                hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)
            if batch % self.hparams.log_interval == 0 and hvd.local_rank() == 0:
                logging.info(self.metric_checker(loss, metrics))
                self.model.reset_metrics()

    def evaluate(self, dataset, epoch=0):
        """ evaluate the model """
        loss_metric = tf.keras.metrics.Mean(name="AverageLoss")
        loss, metrics = None, None
        evaluate_step = self.evaluate_step
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            evaluate_step = tf.function(evaluate_step, input_signature=self.sample_signature)
        self.model.reset_metrics()
        for batch, samples in enumerate(dataset):
            samples = self.model.prepare_samples(samples)
            loss, metrics = evaluate_step(samples)
            if batch % self.hparams.log_interval == 0 and hvd.local_rank() == 0:
                logging.info(self.metric_checker(loss, metrics, -2))
            loss_metric.update_state(loss)
        if hvd.local_rank() == 0:
            logging.info(self.metric_checker(loss_metric.result(), metrics, evaluate_epoch=epoch))
            self.model.reset_metrics()
        return loss_metric.result(), metrics


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)

    json_file = sys.argv[1]
    #config = None
    #with open(json_file) as f:
    #    config = json.load(f)
    #p = parse_config(config)
    HorovodSolver.initialize_devices()
    #multi-servers training should use hvd.rank()
    train(json_file, HorovodSolver, hvd.size(), hvd.rank())
