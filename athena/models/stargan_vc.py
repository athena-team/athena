# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li
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
# Only support eager mode and TF>=2.0.0
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes
""" a implementation of deep speech 2 model can be used as a sample for ctc model """

import tensorflow as tf
from absl import logging
from ..utils.hparam import register_and_parse_hparams
from .base import BaseModel
from ..loss import CTCLoss
from ..loss import StarganLoss
from ..metrics import CTCAccuracy
from ..layers.commons import SUPPORTED_RNNS
from ..models.StarGANVC import StarGANVC

class StarganModel(BaseModel):
    """ a sample implementation of CTC model """
    default_config = {
        "conv_filters": 256,
        "rnn_hidden_size": 1024,
        "num_rnn_layers": 6,
        "rnn_type": "gru"
    }
    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.num_classes = data_descriptions.num_class + 1
        self.loss_function = StarganLoss(blank_index=-1)
        # self.metric = CTCAccuracy()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.StarGAN = StarGANVC()



    def call(self, samples, training=None):
        """ call function """

        # return self.net(samples["input"], training=training)
        return self.StarGAN(samples)

    def compute_logit_length(self, samples):
        """ used for get logit length """
        input_length = tf.cast(samples["input_length"], tf.float32)
        logit_length = tf.math.ceil(input_length / 2)
        logit_length = tf.math.ceil(logit_length / 2)
        logit_length = tf.cast(logit_length, tf.int32)
        return logit_length
