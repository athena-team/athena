# coding=utf-8
# Copyright (C) ATHENA AUTHORS
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
# pylint: disable=unused-argument
""" some metrics """
from absl import logging
import numpy as np
import tensorflow as tf
from .utils.misc import validate_seqs


class CharactorAccuracy:
    """ CharactorAccuracy
    Base class for Word Error Rate calculation
    """

    def __init__(self, name="CharactorAccuracy"):
        self.name = name
        self.inf = tf.constant(np.inf, dtype=tf.float32)
        self.error_count = tf.keras.metrics.Sum(dtype=tf.float32)
        self.total_count = tf.keras.metrics.Sum(dtype=tf.float32)

    def reset_states(self):
        """reset num_err and num_total to zero"""
        self.error_count.reset_states()
        self.total_count.reset_states()

    def update_state(self, sparse_predictions, samples, logit_length=None):
        """ Accumulate errors and counts """
        validated_label = tf.cast(
            tf.sparse.from_dense(samples["output"]), dtype=tf.int64
        )
        labels_counter = tf.cast(tf.shape(validated_label.values)[0], tf.float32)

        num_errs = tf.edit_distance(
            sparse_predictions, validated_label, normalize=False
        )
        num_errs = tf.reduce_sum(num_errs)
        self.error_count(num_errs)
        self.total_count(labels_counter)
        return num_errs, labels_counter

    def __call__(self, logits, samples, logit_length=None):
        return self.update_state(logits, samples, logit_length)

    def result(self):
        """ returns word-error-rate calculated as num_err/num_total """
        error_rate = tf.cast(
            self.error_count.result() / self.total_count.result(), tf.float32
        )
        if error_rate == self.inf:
            return 0.0
        return 1.0 - error_rate


class Seq2SeqSparseCategoricalAccuracy(CharactorAccuracy):
    """ Seq2SeqSparseCategoricalAccuracy
    Inherits CharactorAccuracy and implements Attention accuracy calculation
    """

    def __init__(self, eos, name="Seq2SeqSparseCategoricalAccuracy"):
        super().__init__(name=name)
        self.eos = eos

    def __call__(self, logits, samples, logit_length=None):
        """ Accumulate errors and counts """
        predictions = tf.argmax(logits, axis=2, output_type=tf.int64)
        validated_preds, _ = validate_seqs(predictions, self.eos)

        self.update_state(validated_preds, samples, logit_length)


class CTCAccuracy(CharactorAccuracy):
    """ CTCAccuracy
    Inherits CharactorAccuracy and implements CTC accuracy calculation
    """

    def __init__(self, name="CTCAccuracy"):
        super().__init__(name=name)
        self.need_logit_length = True

    def __call__(self, logits, samples, logit_length=None):
        """ Accumulate errors and counts, logit_length is the output length of encoder"""
        assert logit_length is not None
        with tf.device("/cpu:0"):
            # this only implemented in cpu
            # ignore if the input length is larger than the output length
            if tf.shape(logits)[1] <= tf.shape(samples["output"])[1] + 1:
                logging.warning("the length of logits is shorter than that of labels")
            else:
                decoded, _ = tf.nn.ctc_greedy_decoder(
                    tf.transpose(logits, [1, 0, 2]),
                    tf.cast(logit_length, tf.int32),
                )

                self.update_state(decoded[0], samples)
