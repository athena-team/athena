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


class Tacotron2Loss(tf.keras.losses.Loss):
    """Tacotron2 Loss
    """

    def __init__(self, model, guided_attn_weight=0.0, regularization_weight=0.0,
                 l1_loss_weight=0.0, attn_sigma=0.4, mask_decoder=False,
                 name="Tacotron2Loss"):
        super().__init__(name=name)
        self.model = model
        self.guided_attn_weight = guided_attn_weight
        self.regularization_weight = regularization_weight
        self.l1_loss_weight = l1_loss_weight
        self.mask_decoder = mask_decoder
        self.attn_sigma = attn_sigma

    def __call__(self, outputs, samples, logit_length=None):
        """
        Args:
            outputs: contain elements below:
                att_ws_stack: shape: [batch, y_steps, x_steps]
        """
        final_loss = {}
        before_outs, after_outs, logits_stack, att_ws_stack = outputs
        # perform masking for padded values
        output_length = samples["output_length"]
        output = samples["output"]
        y_steps = tf.shape(output)[1]
        batch = tf.shape(output)[0]
        feat_dim = tf.shape(output)[2]
        if self.mask_decoder:
            mask = tf.sequence_mask(
                output_length, y_steps, dtype=tf.float32
            )
            mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, feat_dim])
        else:
            mask = tf.ones_like(output)
        total_size = tf.cast(tf.reduce_sum(mask), dtype=tf.float32)
        if self.l1_loss_weight > 0:
            l1_loss = tf.abs(after_outs - output) + tf.abs(before_outs - output)
            l1_loss *= mask
            final_loss['l1_loss'] = tf.reduce_sum(l1_loss) / total_size * \
                                    self.l1_loss_weight
        mse_loss = tf.square(after_outs - output) + tf.square(before_outs - output)
        mse_loss *= mask

        indexes = tf.tile(tf.range(y_steps)[tf.newaxis, :], [batch, 1]) # [batch, y_steps]
        end_index = tf.tile((output_length - 1)[:, tf.newaxis], [1, y_steps])
        zeroes = tf.zeros_like(indexes, dtype=tf.float32)
        ones = tf.ones_like(indexes, dtype=tf.float32)
        labels = tf.where(end_index <= indexes, ones, zeroes) # [batch, y_steps]
        # bce_loss is used for stop token prediction
        bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits_stack[:, :, 0])
        bce_loss = bce_loss[:, :, tf.newaxis]
        bce_loss *= mask
        final_loss['mse_loss'] = tf.reduce_sum(mse_loss) / total_size
        final_loss['bce_loss'] = tf.reduce_sum(bce_loss) / total_size

        input_length = samples["input_length"]
        if self.guided_attn_weight > 0:
            # guided_attn_masks shape: [batch_size, y_steps, x_steps]
            attn_masks = self._create_attention_masks(input_length, output_length)
            # length_masks shape: [batch_size, y_steps, x_steps]
            length_masks = self._create_length_masks(input_length, output_length)
            att_ws_stack = tf.cast(att_ws_stack, dtype=tf.float32)
            losses = attn_masks * att_ws_stack
            losses *= length_masks
            loss = tf.reduce_sum(losses)
            total_size = tf.cast(tf.reduce_sum(length_masks), dtype=tf.float32)
            final_loss['guided_attn_loss'] = self.guided_attn_weight * loss / total_size
        if self.regularization_weight > 0:
            computed_vars = [var for var in self.model.trainable_variables
                             if 'bias' not in var.name and \
                                'projection' not in var.name and \
                                'e_function' not in var.name and \
                                'embedding' not in var.name and \
                                'rnn' not in var.name and \
                                'zone_out' not in var.name]
            final_loss['regularization_loss'] = tf.add_n([tf.nn.l2_loss(v) for v in computed_vars]) * \
                          self.regularization_weight

        return final_loss

    def _create_attention_masks(self, input_length, output_length):
        """masks created by attention location

        Args:
            input_length: shape: [batch_size]
            output_length: shape: [batch_size]

        Returns:
            masks: shape: [batch_size, y_steps, x_steps]
        """
        batch_size = tf.shape(input_length)[0]
        input_max_len = tf.reduce_max(input_length)
        output_max_len = tf.reduce_max(output_length)

        # grid_x shape: [output_max_len, input_max_len]
        grid_x, grid_y = tf.meshgrid(tf.range(output_max_len),
                                     tf.range(input_max_len),
                                     indexing='ij')

        grid_x = tf.tile(grid_x[tf.newaxis, :, :], [batch_size, 1, 1])
        grid_y = tf.tile(grid_y[tf.newaxis, :, :], [batch_size, 1, 1])
        input_length = input_length[:, tf.newaxis, tf.newaxis]
        output_length = output_length[:, tf.newaxis, tf.newaxis]
        # masks shape: [batch_size, y_steps, x_steps]
        masks = 1.0 - tf.math.exp(-(grid_y / input_length - grid_x / output_length) ** 2
                                 / (2 * (self.attn_sigma ** 2)))
        masks = tf.cast(masks, dtype=tf.float32)
        return masks

    def _create_length_masks(self, input_length, output_length):
        """masks created by input and output length

        Args:
            input_length: shape: [batch_size]
            output_length: shape: [batch_size]

        Returns:
            masks: shape: [batch_size, output_length, input_length]

        Examples:
            output_length: [6, 8]
            input_length: [3, 5]
            masks:
               [[[1, 1, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [1, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1]]]
        """
        input_max_len = tf.reduce_max(input_length)
        output_max_len = tf.reduce_max(output_length)
        # input_masks shape: [batch_size, max_input_length]
        input_masks = tf.sequence_mask(input_length)
        input_masks = tf.tile(tf.expand_dims(input_masks, 1), [1, output_max_len, 1])
        # output_masks shape: [batch_size, max_output_length]
        output_masks = tf.sequence_mask(output_length)
        output_masks = tf.tile(tf.expand_dims(output_masks, -1), [1, 1, input_max_len])
        masks = tf.math.logical_and(input_masks, output_masks)
        masks = tf.cast(masks, dtype=tf.float32)
        return masks