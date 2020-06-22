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
        loss = tf.reduce_sum(tf.abs(loss, name="L1_loss"), axis=-1)
        loss = tf.reduce_mean(loss)
        return loss

def GeneratorLoss(inputs):
    lambda_cycle = 10
    lambda_identity = 5
    lambda_classifier = 3
    discirmination, input_real, generated_back,identity_map,target_label_reshaped,domain_out_real = inputs
    # ============================Domain classify loss=============================
    domain_real_loss = lambda_classifier * \
                            tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_label_reshaped,
                                                                                      logits=domain_out_real))
    # ============================Generator loss================================
    # Cycle loss
    cycle_loss = tf.reduce_mean(tf.abs(input_real - generated_back))
    identity_loss = lambda_identity * tf.reduce_mean(tf.abs(input_real - identity_map))
    # combine discriminator and generator

    generator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(discirmination), logits=discirmination))

    generator_loss_all = generator_loss + lambda_cycle * cycle_loss + \
                              identity_loss + \
                              domain_real_loss
    return generator_loss_all


def DiscriminatorLoss( inputs):
    discrimination_real, discirmination_fake, target_label_reshaped, domain_out_fake = inputs

    discrimination_real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(discrimination_real),
                                                logits=discrimination_real))

    discrimination_fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(discirmination_fake),
                                                logits=discirmination_fake))
    # _gradient_penalty = 10.0 * tf.square(tf.norm(gradients[0], ord=2) - 1.0)

    domain_fake_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_label_reshaped, logits=domain_out_fake))

    discrimator_loss = discrimination_fake_loss + discrimination_real_loss + domain_fake_loss#+_gradient_penalty
    return discrimator_loss

def ClassifyLoss(inputs):
    target_label_reshaped, domain_out_real = inputs
    domain_real_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_label_reshaped, logits=domain_out_real))
    return domain_real_loss

class StarganLoss(tf.python.keras.Layer):
    def __init__(self, **kwargs):
        super(StarganLoss, self).__init__(**kwargs)

    def call(self, outputs,samples, logit_length=None):
        input_real, target_real, target_label, source_label = samples["input_real"], \
                                                              samples["target_real"], samples["target_label"], samples[
                                                                  "source_label"]
        domain_classifier, discirmination, generated_forward, generated_back, \
        domain_out_real, identity_map, discrimination_real, discirmination_fake, \
        domain_out_fake,target_label_reshaped =outputs

        domain_real_loss = ClassifyLoss(target_label_reshaped, domain_out_real)
        generator_loss_all = GeneratorLoss(discirmination, input_real, generated_back,identity_map,
                                           target_label_reshaped,domain_out_real)
        discrimator_loss = DiscriminatorLoss(discrimination_real, discirmination_fake, target_label_reshaped,
                                             domain_out_fake)
        loss_all = domain_real_loss + generator_loss_all + discrimator_loss
        return  loss_all