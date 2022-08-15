# coding=utf-8
# Copyright (C) 2022 ATHENA AUTHORS;
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
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes
import kenlm
from ..base import BaseModel
import tensorflow as tf


class NgramLM(BaseModel):
    def __init__(self, lm_path):
        super(NgramLM, self).__init__()
        self.model = kenlm.LanguageModel(lm_path)
        self.text_featurizer = None

    def forward(self, inputs, inputs_length=None, training: bool = None, ignored_id=[]):

        if self.text_featurizer is None:
            text_list = inputs
        else:
            text_list = self.text_featurizer.decode_to_list(inputs.numpy().tolist(), ignored_id)

        probs = self.score(text_list)

        res = tf.keras.preprocessing.sequence.pad_sequences(
            probs, padding='post', value=0.0, dtype='float32'
        )
        return res

    def score(self, text_list: list = None):
        """
        text_list: list of char
        """
        if len(text_list) > 0 and type(text_list[-1]) is list:
            return [self.score(line) for line in text_list]
        return [prob for prob, _, _ in self.model.full_scores(" ".join(text_list))]




