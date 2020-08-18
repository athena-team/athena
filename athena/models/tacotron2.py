# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Ruixiong Zhang; Lan Yu;
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

"""tacotron2 implementation"""
import tensorflow as tf

from .base import BaseModel
from ..utils.hparam import register_and_parse_hparams
from ..loss import Tacotron2Loss, GuidedAttentionLoss
from ..layers.commons import ZoneOutCell
from ..layers.attention import LocationAttention, StepwiseMonotonicAttention

class Tacotron2(BaseModel):
    """
    An implementation of Tacotron2
    Reference: NATURAL TTS SYNTHESIS BY CONDITIONING WAVENET ON MEL SPECTROGRAM PREDICTIONS
        https://arxiv.org/pdf/1712.05884.pdf
    """
    default_config = {
        "eunits": 512,
        "e_convlayers": 3,
        "ekernels": 5,
        "e_rnnlayers": 1,
        "zoneout_rate": 0.1,
        "dropout_rate": 0.5,
        "dlayers": 2,
        "dunits": 1024,
        "prenet_layers": 2,
        "prenet_units": 256,
        "postnet_layers": 5,
        "postnet_kernel": 5,
        "postnet_filters": 512,
        "reduction_factor": 1,
        "att_dim": 128,
        "att_chans": 32,
        "att_filters": 15,
        "att_scaling": 2.0,
        "guided_attn_weight": 0.0,
        "clip_outputs": False,
        "clip_lower_bound_decay": 0.1,
        "regularization_weight": 0.0,
        "clip_max_value": 4,
        "l1_loss_weight": 0.0,
        "batch_norm_position": "after",
        "mask_decoder": False,
        "pos_weight": 1.0,
        "step_monotonic": False,
        "sma_mode": 'soft',
        "max_output_length": 15,
        "end_prob": 0.5
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        self.num_class = data_descriptions.num_class
        self.eos = self.num_class - 1
        self.feat_dim = data_descriptions.feat_dim
        self.reduction_factor = self.hparams.reduction_factor

        layers = tf.keras.layers
        input_features = layers.Input(shape=data_descriptions.sample_shape["input"],
                                      dtype=tf.float32)
        inner = layers.Embedding(self.num_class, self.hparams.eunits)(input_features)
        attention_loss_function = GuidedAttentionLoss(self.hparams.guided_attn_weight,
                                                      self.reduction_factor)
        self.loss_function = Tacotron2Loss(self,
                                           attention_loss_function,
                                           regularization_weight=self.hparams.regularization_weight,
                                           l1_loss_weight=self.hparams.l1_loss_weight,
                                           mask_decoder=self.hparams.mask_decoder,
                                           pos_weight=self.hparams.pos_weight)
        self.metric = tf.keras.metrics.Mean(name="AverageLoss")

        # encoder definition
        for _ in tf.range(self.hparams.e_convlayers):
            inner = layers.Conv1D(
                filters=self.hparams.eunits,
                kernel_size=self.hparams.ekernels,
                strides=1,
                padding="same",
                use_bias=False,
                data_format="channels_last",
            )(inner)
            if self.hparams.batch_norm_position == 'after':
                inner = layers.ReLU()(inner)
                inner = layers.BatchNormalization()(inner)
            else:
                inner = layers.BatchNormalization()(inner)
                inner = layers.ReLU()(inner)
            inner = layers.Dropout(self.hparams.dropout_rate)(inner)
        for _ in tf.range(self.hparams.e_rnnlayers):
            forward_layer = layers.RNN(cell=ZoneOutCell(
                units=self.hparams.eunits // 2,
                zoneout_rate=self.hparams.zoneout_rate
                ), return_sequences=True)
            backward_layer = layers.RNN(cell=ZoneOutCell(
                units=self.hparams.eunits // 2,
                zoneout_rate=self.hparams.zoneout_rate
                ), return_sequences=True, go_backwards=True)
            inner = layers.Bidirectional(forward_layer,
                                         backward_layer=backward_layer)(inner)

        self.encoder = tf.keras.Model(inputs=input_features, outputs=inner, name="enc")
        print(self.encoder.summary())

        # rnn definitions in decoder
        self.decoder_rnns = []
        for _ in tf.range(self.hparams.dlayers):
            rnn_cell = ZoneOutCell(
                units=self.hparams.dunits,
                zoneout_rate=self.hparams.zoneout_rate
            )
            rnn_layer = layers.RNN(cell=rnn_cell, return_sequences=True)
            self.decoder_rnns.append(rnn_layer)

        # prenet definition
        input_features_prenet = layers.Input(shape=[None, self.feat_dim * self.reduction_factor],
                                             dtype=tf.float32)
        inner = input_features_prenet
        for _ in tf.range(self.hparams.prenet_layers):
            inner = layers.Dense(self.hparams.prenet_units)(inner)
            inner = layers.ReLU()(inner)
            inner = layers.Dropout(self.hparams.dropout_rate)(inner)
        self.prenet = tf.keras.Model(inputs=input_features_prenet, outputs=inner, name="prenet")
        print(self.prenet.summary())

        if self.hparams.step_monotonic:
            self.attn = StepwiseMonotonicAttention(self.hparams.att_dim,
                                                   self.hparams.att_chans,
                                                   self.hparams.att_filters,
                                                   score_bias_init=2.5,
                                                   mode=self.hparams.sma_mode)
        else:
            self.attn = LocationAttention(self.hparams.att_dim,
                                          self.hparams.att_chans,
                                          self.hparams.att_filters,
                                          scaling=self.hparams.att_scaling)

        # postnet definition
        input_features_postnet = layers.Input(shape=tf.TensorShape([None, self.feat_dim]),
                                              dtype=tf.float32)
        inner = input_features_postnet
        for _ in tf.range(self.hparams.postnet_layers):
            filters = self.hparams.postnet_filters
            inner = layers.Conv1D(filters=filters,
                                  kernel_size=self.hparams.postnet_kernel,
                                  strides=1,
                                  padding="same",
                                  use_bias=False,
                                  data_format="channels_last")(inner)
            if self.hparams.batch_norm_position == 'after':
                inner = tf.nn.tanh(inner)
                inner = layers.BatchNormalization()(inner)
            else:
                inner = layers.BatchNormalization()(inner)
                inner = tf.nn.tanh(inner)
            inner = layers.Dropout(self.hparams.dropout_rate)(inner)
        inner = layers.Dense(self.feat_dim, name='projection')(inner)
        self.postnet = tf.keras.Model(inputs=input_features_postnet, outputs=inner, name="postnet")
        print(self.postnet.summary())

        # projection layers definition
        self.feat_out = layers.Dense(self.feat_dim * self.reduction_factor,
                                     use_bias=False, name='feat_projection')
        self.prob_out = layers.Dense(self.reduction_factor, name='prob_projection')

    def _pad_and_reshape(self, outputs, ori_lens, reverse=False):
        '''
        Args:
            outputs: true labels, shape: [batch, y_steps, feat_dim]
            ori_lens: scalar
        Returns:
            reshaped_outputs: it has to be reshaped to match reduction_factor
                shape: [batch, y_steps / reduction_factor, feat_dim * reduction_factor]
        '''
        batch = tf.shape(outputs)[0]
        y_steps = tf.shape(outputs)[1]
        if reverse:
            routputs = tf.reshape(outputs, [batch, y_steps * self.reduction_factor, -1])
            reshaped_outputs = routputs[:, :ori_lens, :]
            return reshaped_outputs

        remainder = ori_lens % self.reduction_factor
        if remainder == 0:
            return tf.reshape(outputs, [batch, ori_lens // self.reduction_factor,
                                        self.feat_dim * self.reduction_factor])
        padding_lens = self.reduction_factor - remainder
        new_lens = padding_lens + ori_lens
        paddings = tf.zeros([batch, padding_lens, self.feat_dim])
        padding_outputs = tf.concat([outputs, paddings], axis=1)
        return tf.reshape(padding_outputs, [batch, new_lens // self.reduction_factor,
                                            self.feat_dim * self.reduction_factor])

    def call(self, samples, training: bool = None):
        x0 = samples["input"]
        input_length = samples["input_length"]
        encoder_output = self.encoder(x0, training=training) # shape: [batch, x_steps, eunits]
        y0 = samples['output']
        ori_lens = tf.shape(samples['output'])[1]
        if self.reduction_factor > 1:
            y0 = self._pad_and_reshape(samples['output'], ori_lens)
        y_steps = tf.shape(y0)[1]
        y0 = self.initialize_input_y(y0)
        prev_rnn_states, prev_attn_weight, prev_context = \
            self.initialize_states(encoder_output, input_length)

        accum_attn_weight = prev_attn_weight
        outs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        logits = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        attn_weights = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        out, logit, prev_rnn_states, prev_attn_weight, prev_context = \
            self.time_propagate(encoder_output,
                                input_length,
                                y0[:, 0, :],
                                prev_rnn_states,
                                accum_attn_weight,
                                prev_attn_weight,
                                prev_context,
                                training=training)

        outs = outs.write(0, out)
        logits = logits.write(0, logit)
        attn_weights = attn_weights.write(0, prev_attn_weight)
        y_steps = tf.shape(y0)[1]
        accum_attn_weight += prev_attn_weight
        for y_index in tf.range(1, y_steps):
            out, logit, prev_rnn_states, new_weight, prev_context = \
                self.time_propagate(encoder_output,
                                    input_length,
                                    y0[:, y_index, :],
                                    prev_rnn_states,
                                    accum_attn_weight,
                                    prev_attn_weight,
                                    prev_context,
                                    training=training)

            outs = outs.write(y_index, out)
            logits = logits.write(y_index, logit)
            attn_weights = attn_weights.write(y_index, new_weight)
            accum_attn_weight += new_weight
            prev_attn_weight = new_weight
        logits_stack = tf.transpose(logits.stack(), [1, 0, 2]) # [batch, y_steps, reduction_factor]
        logits_stack = self._pad_and_reshape(logits_stack, ori_lens, reverse=True)
        before_outs = tf.transpose(outs.stack(), [1, 0, 2]) # [batch, y_steps, feat_dim]
        before_outs = self._pad_and_reshape(before_outs, ori_lens, reverse=True)
        if self.hparams.clip_outputs:
            maximum = - self.hparams.clip_max_value - self.hparams.clip_lower_bound_decay
            maximum = tf.maximum(before_outs, maximum)
            before_outs = tf.minimum(maximum, self.hparams.clip_max_value)
        # attn_weights_stack, shape: # [batch, y_steps, x_steps]
        attn_weights_stack = tf.transpose(attn_weights.stack(), [1, 0, 2])
        # after_outs, shape: [batch, y_steps, feat_dim]
        after_outs = before_outs + self.postnet(before_outs, training=training)
        if self.hparams.clip_outputs:
            maximum = - self.hparams.clip_max_value - self.hparams.clip_lower_bound_decay
            maximum = tf.maximum(after_outs, maximum)
            after_outs = tf.minimum(maximum, self.hparams.clip_max_value)

        return before_outs, after_outs, logits_stack, attn_weights_stack

    def initialize_input_y(self, y):
        """
        Args:
            y: the true label, shape: [batch, y_steps, feat_dim]
        Returns:
            y0: zeros will be padded as one step to the start step
                shape: [batch, y_steps+1, feat_dim]
        """
        batch = tf.shape(y)[0]
        prev_out = tf.zeros([batch, 1, self.feat_dim * self.reduction_factor])
        return tf.concat([prev_out, y], axis=1)[:, :-1, :]

    def initialize_states(self, encoder_output, input_length):
        """
        Args:
            encoder_output: encoder outputs, shape: [batch, x_step, eunits]
            input_length: shape: [batch]
        Returns:
            prev_rnn_states: initial states of rnns in decoder
                [rnn layers, 2, batch, dunits]
            prev_attn_weight: initial attention weights, [batch, x_steps]
            prev_context: initial context, [batch, eunits]
        """
        # initialize memory states of rnn
        batch = tf.shape(encoder_output)[0]
        prev_rnn_states = [(tf.zeros([batch, self.hparams.dunits]),
                            tf.zeros([batch, self.hparams.dunits]))
                           for _ in range(len(self.decoder_rnns))]
        x_steps = tf.shape(encoder_output)[1]
        prev_attn_weight = self.attn.initialize_weights(input_length, x_steps)
        # initialize context
        prev_context = tf.zeros([batch, self.hparams.eunits])

        return prev_rnn_states, prev_attn_weight, prev_context

    def time_propagate(self, encoder_output, input_length, prev_y,
                       prev_rnn_states, accum_attn_weight, prev_attn_weight, prev_context,
                       training=False):
        """
        Args:
            encoder_output: encoder output (batch, x_steps, eunits).
            input_length: (batch,)
            prev_y: one step of true labels or predicted labels (batch, feat_dim).
            prev_rnn_states: previous rnn states [layers, 2, states] for lstm
            prev_attn_weight: previous attention weights, shape: [batch, x_steps]
            prev_context: previous context vector: [batch, attn_dim]
            training: if it is training mode
        Returns:
            out: shape: [batch, feat_dim]
            logit: shape: [batch, reduction_factor]
            current_rnn_states: [rnn_layers, 2, batch, dunits]
            attn_weight: [batch, x_steps]
        """
        prenet_out = self.prenet(prev_y, training=training)
        decode_s = tf.concat([prev_context, prenet_out], axis=1)
        current_rnn_states = []
        _, rnn_states = self.decoder_rnns[0].cell(decode_s,
                                                  prev_rnn_states[0],
                                                  training=training)
        current_rnn_states.append(rnn_states)
        for l, rnn_layer in enumerate(self.decoder_rnns[1:]):
            _, new_states = rnn_layer.cell(current_rnn_states[l][0],
                                           prev_rnn_states[l + 1],
                                           training=training)
            current_rnn_states.append(new_states)
        decode_s = current_rnn_states[-1][0]
        attn_inputs = (encoder_output, input_length)
        prev_states = (decode_s, accum_attn_weight, prev_attn_weight)
        context, attn_weight = self.attn(attn_inputs, prev_states, training=training)
        rnn_output = tf.concat([decode_s, context], axis=1)
        out = self.feat_out(rnn_output)
        logit = self.prob_out(rnn_output)
        return out, logit, current_rnn_states, attn_weight, context

    def get_loss(self, outputs, samples, training=None):
        loss = self.loss_function(outputs, samples)
        total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
        self.metric.update_state(total_loss)
        metrics = {self.metric.name: self.metric.result()}
        return loss, metrics

    def synthesize(self, samples):
        """
        Synthesize acoustic features from the input texts
        Args:
            samples: the data source to be synthesized
        Returns:
            after_outs: the corresponding synthesized acoustic features
            attn_weights_stack: the corresponding attention weights
        """
        x0 = samples["input"]
        input_length = samples["input_length"]
        batch = tf.shape(x0)[0]
        encoder_output = self.encoder(x0, training=False) # shape: [batch, x_steps, eunits]
        prev_rnn_states, prev_attn_weight, prev_context = \
            self.initialize_states(encoder_output, input_length)
        accum_attn_weight = prev_attn_weight
        outs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        logits = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        attn_weights = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        initial_y = tf.zeros([batch, self.feat_dim * self.reduction_factor])
        out, logit, prev_rnn_states, prev_attn_weight, prev_context = \
            self.time_propagate(encoder_output,
                                input_length,
                                initial_y,
                                prev_rnn_states,
                                accum_attn_weight,
                                prev_attn_weight,
                                prev_context,
                                training=False)

        outs = outs.write(0, out)
        logits = logits.write(0, logit)
        attn_weights = attn_weights.write(0, prev_attn_weight)
        y_index = 0
        accum_attn_weight += prev_attn_weight
        max_output_len = self.hparams.max_output_length * input_length[0] // self.reduction_factor
        for _ in tf.range(max_output_len):
            y_index += 1
            out, logit, prev_rnn_states, new_weight, prev_context = \
                self.time_propagate(encoder_output,
                                    input_length,
                                    out,
                                    prev_rnn_states,
                                    accum_attn_weight,
                                    prev_attn_weight,
                                    prev_context,
                                    training=False)

            outs = outs.write(y_index, out)
            logits = logits.write(y_index, logit)
            attn_weights = attn_weights.write(y_index, new_weight)
            prev_attn_weight = new_weight
            accum_attn_weight += new_weight
            probs = tf.nn.sigmoid(logit)
            time_to_end = probs > self.hparams.end_prob
            time_to_end = tf.reduce_any(time_to_end)
            if time_to_end:
                break

        logits_stack = tf.transpose(logits.stack(), [1, 0, 2]) # [batch, y_steps, reduction_factor]
        # before_outs: [batch, y_steps, feat_dim*reduction_factor]
        before_outs = tf.transpose(outs.stack(), [1, 0, 2])
        attn_weights_stack = tf.transpose(attn_weights.stack(), [1, 0, 2])
        after_outs = self._synthesize_post_net(before_outs, logits_stack)
        return after_outs, attn_weights_stack

    def _synthesize_post_net(self, before_outs, logits_stack):
        if self.hparams.clip_outputs:
            maximum = - self.hparams.clip_max_value - self.hparams.clip_lower_bound_decay
            maximum = tf.maximum(before_outs, maximum)
            before_outs = tf.minimum(maximum, self.hparams.clip_max_value)
        output_lens = tf.shape(before_outs)[1] * self.reduction_factor
        padded_logits = self._pad_and_reshape(logits_stack, output_lens, reverse=True)
        # paddings should be discarded or the end of the wav may contains noises
        end_prob_num = padded_logits[:, -self.reduction_factor:, :] > self.hparams.end_prob
        end_prob_num = tf.cast(end_prob_num, dtype=tf.int32)
        end_prob_num = tf.reduce_sum(end_prob_num)
        real_lens = output_lens - end_prob_num + 1
        before_outs = self._pad_and_reshape(before_outs, real_lens, reverse=True)
        # after_outs: [batch, y_steps, feat_dim]
        after_outs = before_outs + self.postnet(before_outs, training=False)
        if self.hparams.clip_outputs:
            maximum = - self.hparams.clip_max_value - self.hparams.clip_lower_bound_decay
            maximum = tf.maximum(after_outs, maximum)
            after_outs = tf.minimum(maximum, self.hparams.clip_max_value)
        return after_outs
