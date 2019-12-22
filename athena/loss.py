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
# pylint: disable=too-few-public-methods
""" some losses """
import tensorflow as tf
from .utils.misc import insert_eos_in_labels


class CTCLoss(tf.keras.losses.Loss):
    """ CTC LOSS
	CTC LOSS implemented with Tensorflow
    """

    def __init__(self, logits_time_major=False, blank_index=-1, name="CTCLoss"):
        super().__init__(name=name)
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index
        self.need_logit_length = True

    def __call__(self, logits, samples, logit_length=None):
        assert logit_length is not None
        # use v2 ctc_loss
        ctc_loss = tf.nn.ctc_loss(
            labels=samples["output"],
            logits=logits,
            logit_length=logit_length,
            label_length=samples["output_length"],
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index,
        )
        return tf.reduce_mean(ctc_loss)


class Seq2SeqSparseCategoricalCrossentropy(tf.keras.losses.CategoricalCrossentropy):
    """ Seq2SeqSparseCategoricalCrossentropy LOSS
    CategoricalCrossentropy calculated at each character for each sequence in a batch
    """

    def __init__(self, num_classes, eos=-1, by_token=False, by_sequence=True,
                 from_logits=True, label_smoothing=0.0):
        super().__init__(from_logits=from_logits, label_smoothing=label_smoothing, reduction="none")
        self.by_token = by_token
        self.by_sequence = by_sequence
        self.num_classes = num_classes
        self.eos = num_classes + eos if eos < 0 else eos

    def __call__(self, logits, samples, logit_length=None):
        labels = insert_eos_in_labels(samples["output"], self.eos, samples["output_length"])
        mask = tf.math.logical_not(tf.math.equal(labels, 0))
        labels = tf.one_hot(indices=labels, depth=self.num_classes)
        seq_len = tf.shape(labels)[1]
        logits = logits[:, :seq_len, :]
        loss = self.call(labels, logits)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        if self.by_token:
            return tf.divide(tf.reduce_sum(loss), tf.reduce_sum(mask))
        if self.by_sequence:
            loss = tf.reduce_sum(loss, axis=-1)
        return tf.reduce_mean(loss)


class MPCLoss(tf.keras.losses.Loss):
    """MPC LOSS
    L1 loss for each masked acoustic features in a batch
    """

    def __init__(self, name="MPCLoss"):
        super().__init__(name=name)

    def __call__(self, logits, samples, logit_length=None):
        target = samples["output"]
        shape = tf.shape(logits)
        target = tf.reshape(target, shape)
        loss = target - logits
        # mpc mask
        mask = tf.cast(tf.math.equal(tf.reshape(samples["input"], shape), 0), loss.dtype)
        loss *= mask
        # sequence length mask
        seq_mask = tf.sequence_mask(logit_length, shape[1], dtype=loss.dtype)
        seq_mask = tf.tile(seq_mask[:, :, tf.newaxis], [1, 1, shape[2]])
        loss *= seq_mask
        loss = tf.reduce_sum(tf.abs(loss, name="L1_loss"), 2)
        loss = tf.reduce_mean(loss)
        return loss
