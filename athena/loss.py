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
# pylint: disable=too-few-public-methods, too-many-arguments
""" some losses """
import math
import tensorflow as tf
from .utils.misc import insert_eos_in_labels, apply_label_smoothing


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
        super().__init__(from_logits=from_logits, reduction="none")
        self.by_token = by_token
        self.by_sequence = by_sequence
        self.num_classes = num_classes
        self.eos = num_classes + eos if eos < 0 else eos
        self.smoothing_rate = label_smoothing

    def __call__(self, logits, samples, logit_length=None):
        labels = insert_eos_in_labels(samples["output"], self.eos, samples["output_length"])
        mask = tf.math.logical_not(tf.math.equal(labels, 0))
        labels = tf.one_hot(indices=labels, depth=self.num_classes)
        if self.smoothing_rate != 0.0:
            labels = apply_label_smoothing(labels, 
                num_classes=self.num_classes, 
                smoothing_rate=self.smoothing_rate)
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
    """ MPC LOSS
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
    """ Tacotron2 Loss
    """

    def __init__(self, model, guided_attn_loss_function, regularization_weight=0.0,
                 l1_loss_weight=0.0, mask_decoder=False, pos_weight=1.0,
                 name="Tacotron2Loss"):
        super().__init__(name=name)
        self.model = model
        self.regularization_weight = regularization_weight
        self.l1_loss_weight = l1_loss_weight
        self.mask_decoder = mask_decoder
        self.guided_attn_loss_function = guided_attn_loss_function
        self.pos_weight = pos_weight

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
        bce_loss = tf.nn.weighted_cross_entropy_with_logits(labels=labels,
                                                            logits=logits_stack[:, :, 0],
                                                            pos_weight=self.pos_weight)
        bce_loss = bce_loss[:, :, tf.newaxis]
        bce_loss *= mask
        final_loss['mse_loss'] = tf.reduce_sum(mse_loss) / total_size
        final_loss['bce_loss'] = tf.reduce_sum(bce_loss) / total_size
        if self.guided_attn_loss_function is not None and \
                self.guided_attn_loss_function.guided_attn_weight > 0:
            final_loss['guided_attn_loss'] = self.guided_attn_loss_function(att_ws_stack, samples)
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


class GuidedAttentionLoss(tf.keras.losses.Loss):
    """ GuidedAttention Loss to make attention alignments more monotonic
    """

    def __init__(self, guided_attn_weight, reduction_factor, attn_sigma=0.4,
                 name='GuidedAttentionLoss'):
        super().__init__(name=name)
        self.guided_attn_weight = guided_attn_weight
        self.reduction_factor = reduction_factor
        self.attn_sigma = attn_sigma

    def __call__(self, att_ws_stack, samples):
        output_length = samples["output_length"]
        input_length = samples["input_length"]
        reduction_output_length = (output_length - 1) // self.reduction_factor + 1
        # attn_masks shape: [batch_size, 1, reduction_y_steps, x_steps]
        attn_masks = self._create_attention_masks(input_length, reduction_output_length)
        # length_masks shape: [batch_size, 1, reduction_y_steps, x_steps]
        length_masks = self._create_length_masks(input_length, reduction_output_length)
        att_ws_stack = tf.cast(att_ws_stack, dtype=tf.float32)
        if len(tf.shape(att_ws_stack)) == 3:
            att_ws_stack = tf.expand_dims(att_ws_stack, axis=1)
        losses = attn_masks * att_ws_stack
        losses *= length_masks
        loss = tf.reduce_sum(losses)
        total_size = tf.cast(tf.reduce_sum(length_masks), dtype=tf.float32)
        return self.guided_attn_weight * loss / total_size

    def _create_attention_masks(self, input_length, output_length):
        """masks created by attention location

        Args:
            input_length: shape: [batch_size]
            output_length: shape: [batch_size]

        Returns:
            masks: shape: [batch_size, 1, y_steps, x_steps]
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
        masks = tf.expand_dims(masks, axis=1)
        masks = tf.cast(masks, dtype=tf.float32)
        return masks

    def _create_length_masks(self, input_length, output_length):
        """masks created by input and output length

        Args:
            input_length: shape: [batch_size]
            output_length: shape: [batch_size]

        Returns:
            masks: shape: [batch_size, 1, output_length, input_length]

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
        masks = tf.expand_dims(masks, axis=1)
        masks = tf.cast(masks, dtype=tf.float32)
        return masks


class GuidedMultiHeadAttentionLoss(GuidedAttentionLoss):
    """Guided multihead attention loss function module for multi head attention."""

    def __init__(self, guided_attn_weight, reduction_factor, attn_sigma=0.4, num_heads=2,
                 num_layers=2, name='GuidedMultiHeadAttentionLoss'):
        super().__init__(guided_attn_weight, reduction_factor, attn_sigma, name=name)
        self.num_heads = num_heads
        self.num_layers = num_layers

    def __call__(self, att_ws_stack, samples):
        # att_ws_layer: shape: [batch, head, y_steps, x_steps]
        total_loss = 0
        total_layers = len(att_ws_stack)
        for index, layer_index in enumerate(reversed(range(total_layers))):
            if index >= self.num_layers:
                break
            att_ws_layer = att_ws_stack[layer_index]
            total_loss += super().__call__(att_ws_layer[:, :self.num_heads], samples)
        return total_loss


class FastSpeechLoss(tf.keras.losses.Loss):
    """used for training of fastspeech"""

    def __init__(self, duration_predictor_loss_weight, eps=1.0, use_mask=True, teacher_guide=False):
        super().__init__()
        self.eps = eps
        self.duration_predictor_loss_weight = duration_predictor_loss_weight
        self.use_mask = use_mask
        self.teacher_guide = teacher_guide

    def __call__(self, outputs, samples):
        """
        Its corresponding log value is calculated to make it Gaussian.
        Args:
            outputs: it contains four elements:
                before_outs: outputs before postnet, shape: [batch, y_steps, feat_dim]
                teacher_outs: teacher outputs, shape: [batch, y_steps, feat_dim]
                after_outs: outputs after postnet, shape: [batch, y_steps, feat_dim]
                duration_sequences: duration predictions from teacher model, shape: [batch, x_steps]
                pred_duration_sequences: duration predictions from trained predictor
                    shape: [batch, x_steps]
            samples: samples from dataset

        """
        final_loss = {}
        before_outs, teacher_outs, after_outs, duration_sequences, pred_duration_sequences = outputs
        # perform masking for padded values
        output_length = samples["output_length"]
        output = samples["output"] # [batch, y_steps, feat_dim]
        input_length = samples['input_length']
        x_steps = tf.reduce_max(input_length)
        y_steps = tf.shape(output)[1]
        feat_dim = tf.shape(output)[2]
        if self.use_mask:
            mask = tf.sequence_mask(output_length, y_steps, dtype=tf.float32)
            mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, feat_dim])
        else:
            mask = tf.ones_like(output)
        total_size = tf.cast(tf.reduce_sum(mask), dtype=tf.float32)

        if self.teacher_guide:
            l1_loss = tf.abs(after_outs - teacher_outs) + tf.abs(before_outs - teacher_outs)
        else:
            l1_loss = tf.abs(after_outs - output) + tf.abs(before_outs - output)
        l1_loss *= mask
        final_loss['l1_loss'] = tf.reduce_sum(l1_loss) / total_size

        teacher_duration = tf.math.log(tf.cast(duration_sequences, dtype=tf.float32) + self.eps)
        if self.use_mask:
            mask = tf.sequence_mask(input_length, x_steps, dtype=tf.float32)
        else:
            mask = tf.ones_like(teacher_duration)
        total_size = tf.cast(tf.reduce_sum(mask), dtype=tf.float32)
        duration_loss = tf.square(teacher_duration - pred_duration_sequences)
        duration_loss *= mask
        final_loss['duration_loss'] = tf.reduce_sum(duration_loss) / total_size

        return final_loss


class SoftmaxLoss(tf.keras.losses.Loss):
    """ Softmax Loss
        Similar to this implementation "https://github.com/clovaai/voxceleb_trainer"
    """
    def __init__(self, embedding_size, num_classes, name="SoftmaxLoss"):
        super().__init__(name=name)
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.criterion = tf.nn.softmax_cross_entropy_with_logits
        self.dense = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(embedding_size,),
        )

    def __call__(self, outputs, samples, logit_length=None):
        labels = tf.squeeze(samples['output'])
        outputs = self.dense(outputs)
        label_onehot = tf.one_hot(labels, self.num_classes)
        loss = tf.reduce_mean(self.criterion(label_onehot, outputs))
        return loss, outputs


class AMSoftmaxLoss(tf.keras.losses.Loss):
    """ Additive Margin Softmax Loss
        Reference to paper "CosFace: Large Margin Cosine Loss for Deep Face Recognition"
                            and "In defence of metric learning for speaker recognition"
        Similar to this implementation "https://github.com/clovaai/voxceleb_trainer"
    """
    def __init__(self, embedding_size, num_classes, m=0.3, s=15, name="AMSoftmaxLoss"):
        super().__init__(name=name)
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.m = m
        self.s = s
        initializer = tf.initializers.GlorotNormal()
        self.weight = tf.Variable(initializer(
                             shape=[embedding_size, num_classes], dtype=tf.float32),
                             name="AMSoftmaxLoss_weight")
        self.criterion = tf.nn.softmax_cross_entropy_with_logits

    def __call__(self, outputs, samples, logit_length=None):
        labels = tf.squeeze(samples['output'])
        outputs_norm = tf.math.l2_normalize(outputs, axis=1)
        weight_norm = tf.math.l2_normalize(self.weight, axis=0)
        costh = tf.matmul(outputs_norm, weight_norm)

        label_onehot = tf.one_hot(labels, self.num_classes)
        delt_costh = self.m * label_onehot

        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = tf.reduce_mean(self.criterion(label_onehot, costh_m_s))
        return loss, costh_m_s


class AAMSoftmaxLoss(tf.keras.losses.Loss):
    """ Additive Angular Margin Softmax Loss
        Reference to paper "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
                            and "In defence of metric learning for speaker recognition"
        Similar to this implementation "https://github.com/clovaai/voxceleb_trainer"
    """
    def __init__(self, embedding_size, num_classes,
                 m=0.3, s=15, easy_margin=False, name="AAMSoftmaxLoss"):
        super().__init__(name=name)
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.m = m
        self.s = s
        initializer = tf.initializers.GlorotNormal()
        self.weight = tf.Variable(initializer(
                             shape=[embedding_size, num_classes], dtype=tf.float32),
                             name="AAMSoftmaxLoss_weight")
        self.criterion = tf.nn.softmax_cross_entropy_with_logits
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def __call__(self, outputs, samples, logit_length=None):
        labels = tf.squeeze(samples['output'])
        outputs_norm = tf.math.l2_normalize(outputs, axis=1)
        weight_norm = tf.math.l2_normalize(self.weight, axis=0)
        cosine = tf.matmul(outputs_norm, weight_norm)
        sine = tf.clip_by_value(tf.math.sqrt(1.0 - tf.math.pow(cosine, 2)), 0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where((cosine - self.th) > 0, phi, cosine - self.mm)

        label_onehot = tf.one_hot(labels, self.num_classes)
        output = (label_onehot * phi) + ((1.0 - label_onehot) * cosine)
        output = output * self.s
        loss = tf.reduce_mean(self.criterion(label_onehot, output))
        return loss, output


class ProtoLoss(tf.keras.losses.Loss):
    """ Prototypical Loss
        Reference to paper "Prototypical Networks for Few-shot Learning"
                            and "In defence of metric learning for speaker recognition"
        Similar to this implementation "https://github.com/clovaai/voxceleb_trainer"
    """
    def __init__(self, name="ProtoLoss"):
        super().__init__(name=name)
        self.criterion = tf.nn.softmax_cross_entropy_with_logits

    def __call__(self, outputs, samples=None, logit_length=None):
        """
            Args:
                outputs: [batch_size, num_speaker_utts, embedding_size]
        """
        out_anchor = tf.math.reduce_mean(outputs[:, 1:, :], axis=1)
        out_positive = outputs[:, 0, :]
        step_size = tf.shape(out_anchor)[0]

        out_positive_reshape = tf.tile(tf.expand_dims(out_positive, -1), [1, 1, step_size])
        out_anchor_reshape = tf.transpose(tf.tile(
                             tf.expand_dims(out_anchor, -1), [1, 1, step_size]), [2, 1, 0])

        distance = -tf.reduce_sum(tf.math.squared_difference(
                                  out_positive_reshape, out_anchor_reshape), axis=1)
        label = tf.one_hot(tf.range(step_size), step_size)
        loss = tf.reduce_mean(self.criterion(label, distance))
        return loss


class AngleProtoLoss(tf.keras.losses.Loss):
    """ Angular Prototypical Loss
        Reference to paper "In defence of metric learning for speaker recognition"
        Similar to this implementation "https://github.com/clovaai/voxceleb_trainer"
    """
    def __init__(self, init_w=10.0, init_b=-5.0, name="AngleProtoLoss"):
        super().__init__(name=name)
        self.weight = tf.Variable(initial_value=init_w)
        self.bias = tf.Variable(initial_value=init_b)
        self.criterion = tf.nn.softmax_cross_entropy_with_logits
        self.cosine_similarity = tf.keras.losses.cosine_similarity

    def __call__(self, outputs, samples=None, logit_length=None):
        """
             Args:
                outputs: [batch_size, num_speaker_utts, embedding_size]
        """
        out_anchor = tf.math.reduce_mean(outputs[:, 1:, :], axis=1)
        out_positive = outputs[:, 0, :]
        step_size = tf.shape(out_anchor)[0]

        out_positive_reshape = tf.tile(tf.expand_dims(out_positive, -1), [1, 1, step_size])
        out_anchor_reshape = tf.transpose(tf.tile(
                             tf.expand_dims(out_anchor, -1), [1, 1, step_size]), [2, 1, 0])
        cosine = -self.cosine_similarity(out_positive_reshape, out_anchor_reshape, axis=1)
        self.weight = tf.clip_by_value(self.weight, tf.constant(1e-6), tf.float32.max)
        cosine_w_b = self.weight * cosine + self.bias

        labels = tf.one_hot(tf.range(step_size), step_size)
        loss = tf.reduce_mean(self.criterion(labels, cosine_w_b))
        return loss


class GE2ELoss(tf.keras.losses.Loss):
    """ Generalized End-to-end Loss
        Reference to paper "Generalized End-to-end Loss for Speaker Verification"
                            and "In defence of metric learning for speaker recognition"
        Similar to this implementation "https://github.com/clovaai/voxceleb_trainer"
    """
    def __init__(self, init_w=10.0, init_b=-5.0, name="GE2ELoss"):
        super().__init__(name=name)
        self.weight = tf.Variable(initial_value=init_w)
        self.bias = tf.Variable(initial_value=init_b)
        self.criterion = tf.nn.softmax_cross_entropy_with_logits
        self.cosine_similarity = tf.keras.losses.cosine_similarity

    def __call__(self, outputs, samples=None, logit_length=None):
        """
             Args:
                outputs: [batch_size, num_speaker_utts, embedding_size]
        """
        step_size = tf.shape(outputs)[0]
        num_speaker_utts = tf.shape(outputs)[1]
        centroids = tf.math.reduce_mean(outputs, axis=1)
        cosine_all = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        for utt in range(0, num_speaker_utts):
            index = [*range(0, num_speaker_utts)]
            index.remove(utt)
            out_positive = outputs[:, utt, :]
            out_anchor = tf.math.reduce_mean(tf.gather(outputs, tf.constant(index), axis=1), axis=1)

            out_positive_reshape = tf.tile(tf.expand_dims(out_positive, -1), [1, 1, step_size])
            centroids_reshape = tf.transpose(tf.tile(
                                tf.expand_dims(centroids, -1), [1, 1, step_size]), [2, 1, 0])

            cosine_diag = -self.cosine_similarity(out_positive, out_anchor, axis=1)
            cosine = -self.cosine_similarity(out_positive_reshape, centroids_reshape, axis=1)

            cosine_update = tf.linalg.set_diag(cosine, cosine_diag)
            cosine_all.write(utt,
                            tf.clip_by_value(cosine_update, tf.constant(1e-6), tf.float32.max))

        self.weight = tf.clip_by_value(self.weight, tf.constant(1e-6), tf.float32.max)
        cosine_stack = tf.transpose(cosine_all.stack(), perm=[1, 0, 2])
        cosine_w_b = self.weight * cosine_stack + self.bias
        cosine_w_b_reshape = tf.reshape(cosine_w_b, [-1, step_size])

        labels_repeat = tf.reshape(tf.tile(tf.expand_dims(tf.range(step_size), -1),
                                           [1, num_speaker_utts]), [-1])
        labels_onehot = tf.one_hot(labels_repeat, step_size)
        loss = tf.reduce_mean(self.criterion(labels_onehot, cosine_w_b_reshape))
        return loss


class StarganLoss(tf.keras.losses.Loss):
    """ Loss for stargan model, it consists of three parts, generator_loss,
        discriminator_loss and classifier_loss. lambda_identity and lambda_classifier 
        is added to make loss values comparable
    """
    def __init__(self, lambda_cycle, lambda_identity, lambda_classifier, name="StarganLoss"):
        super().__init__(name=name)
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_classifier = lambda_classifier

    def __call__(self, outputs, samples, logit_length=None, stage=None):
        input_real, target_real, target_label, source_label = \
            samples["src_coded_sp"], samples["tar_coded_sp"], \
            samples["tar_speaker"], samples["src_speaker"]

        target_label_reshaped = outputs["target_label_reshaped"]

        if stage == "classifier":
            domain_out_real = outputs["domain_out_real"]
            loss = ClassifyLoss(target_label_reshaped, domain_out_real)
        elif stage == "generator":
            discirmination, generated_back, identity_map, domain_out_real = \
            outputs["discirmination"], outputs["generated_back"], outputs["identity_map"], outputs["domain_out_real"]

            loss = GeneratorLoss(discirmination, input_real, generated_back, identity_map,\
                                           target_label_reshaped, domain_out_real, self.lambda_cycle,\
                                           self.lambda_identity, self.lambda_classifier)
        else:
            discrimination_real, discirmination, domain_out_fake, gradient_penalty = outputs["discrimination_real"], \
                outputs["discirmination"], outputs["domain_out_fake"], outputs["gradient_penalty"]
            loss = DiscriminatorLoss(discrimination_real, discirmination,\
                                             target_label_reshaped, domain_out_fake, gradient_penalty)
        return loss


def GeneratorLoss(discirmination, input_real, generated_back, identity_map, target_label_reshaped,\
                  domain_out_real, lambda_cycle, lambda_identity, lambda_classifier):
    """ Loss for generator part of stargan model, it consists of cycle_loss with unparallel data, 
        identity loss for data coming from same class and another loss from tricking the discriminator
    """
    domain_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits \
                                      (labels=target_label_reshaped, logits=domain_out_real))

    # we need to trim input to have the same length as output because it might not be divisible with 4
    length = tf.math.minimum(tf.shape(generated_back)[2], tf.shape(input_real)[2])
    cycle_loss = tf.reduce_mean(tf.abs(input_real[:, :, 0: length, :] - generated_back[:, :, 0: length, :]))
    identity_loss = tf.reduce_mean(tf.abs(input_real[:, :, 0: length, :] - identity_map[:, :, 0: length, :]))

    generator_loss = tf.reduce_mean(\
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(discirmination),\
                                            logits=discirmination))

    generator_loss_all = generator_loss + lambda_cycle * cycle_loss + \
                         lambda_identity * identity_loss + lambda_classifier * domain_real_loss
    return generator_loss_all


def DiscriminatorLoss(discrimination_real, discirmination_fake, target_label_reshaped,\
                                             domain_out_fake, gradient_penalty):
    """ Loss for discriminator part of stargan model, it consists of discrimination loss from real and 
        generated data and domain classification loss from generated data  
    """
    discrimination_real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(discrimination_real),\
                                                logits=discrimination_real))

    discrimination_fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(discirmination_fake),\
                                                logits=discirmination_fake))

    domain_fake_loss = tf.reduce_mean(\
        tf.nn.softmax_cross_entropy_with_logits(labels=target_label_reshaped, logits=domain_out_fake))

    discrimator_loss = discrimination_fake_loss + discrimination_real_loss + domain_fake_loss \
                        + gradient_penalty
    return discrimator_loss


def ClassifyLoss(target_label_reshaped, domain_out_real):
    """ Loss for classifier part of stargan model, it consists of classifier loss from real data 
    """
    domain_real_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=target_label_reshaped, logits=domain_out_real))
    return domain_real_loss


