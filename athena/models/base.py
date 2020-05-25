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
# Only support eager mode
# pylint: disable=useless-super-delegation, unused-argument, no-self-use

""" base model for models """
from absl import logging
import tensorflow as tf

class BaseModel(tf.keras.Model):
    """Base class for model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = None
        self.metric = None

    def call(self, samples, training=None):
        """ call model """
        raise NotImplementedError()

    #pylint: disable=not-callable
    def get_loss(self, outputs, samples, training=None):
        """ get loss """
        if self.loss_function is None:
            loss = 0.0
        else:
            logit_length = self.compute_logit_length(samples)
            loss = self.loss_function(outputs, samples, logit_length)
        if self.metric is None:
            metrics = {}
        else:
            self.metric(outputs, samples, logit_length)
            metrics = {self.metric.name: self.metric.result()}
        return loss, metrics

    def compute_logit_length(self, samples):
        """ compute the logit length """
        return samples["input_length"]

    def reset_metrics(self):
        """ reset the metrics """
        if self.metric is not None:
            self.metric.reset_states()

    def prepare_samples(self, samples):
        """ for special data prepare
        carefully: do not change the shape of samples
        """
        return samples

    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        """ restore from pretrained model
        """
        logging.info("restore from pretrained model")

    def decode(self, samples, hparams, decoder):
        """ decode interface
        """
        logging.info("sorry, this model do not support decode")
