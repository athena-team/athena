# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Ruixiong Zhang;
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

""" speech transformer implementation"""

import tensorflow as tf
from .tacotron2 import Tacotron2
from ..loss import Tacotron2Loss, GuidedMultiHeadAttentionLoss
from ..layers.commons import ScaledPositionalEncoding
from ..layers.transformer import Transformer
from ..utils.hparam import register_and_parse_hparams
from ..utils.misc import generate_square_subsequent_mask, create_multihead_mask


class TTSTransformer(Tacotron2):
    """ TTS version of SpeechTransformer. Model mainly consists of three parts:
    the x_net for input preparation, the y_net for output preparation and the transformer itself
    Reference: Neural Speech Synthesis with Transformer Network
        (https://arxiv.org/pdf/1809.08895.pdf)
    """

    default_config = {
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 12,
        "num_decoder_layers": 6,
        "dff": 1280,
        "rate": 0.1,

        "dropout_rate": 0.5,
        "prenet_layers": 2,
        "prenet_units": 256,

        "postnet_layers": 5,
        "postnet_kernel": 5,
        "postnet_filters": 512,

        "guided_attn_weight": 0.0,
        "regularization_weight": 0.0,
        "l1_loss_weight": 1.0,
        "batch_norm_position": "after",
        "mask_decoder": False,
        "pos_weight": 1.0,

        "reduction_factor": 3,
        "clip_outputs": False,
        "clip_lower_bound_decay": 0.1,
        "clip_max_value": 4,

        "max_output_length": 15,
        "end_prob": 0.5
    }

    def __init__(self, data_descriptions, config=None):
        self.default_config.update(config)
        current_default_config = self.default_config
        self.default_config = super().default_config
        self.default_config.update(current_default_config)
        super().__init__(data_descriptions)
        del self.encoder
        del self.decoder_rnns
        del self.attn
        self.default_config = current_default_config
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        attention_loss_function = GuidedMultiHeadAttentionLoss(self.hparams.guided_attn_weight,
                                                               self.reduction_factor)
        self.loss_function = Tacotron2Loss(self,
                                           attention_loss_function,
                                           regularization_weight=self.hparams.regularization_weight,
                                           l1_loss_weight=self.hparams.l1_loss_weight,
                                           mask_decoder=self.hparams.mask_decoder,
                                           pos_weight=self.hparams.pos_weight)

        # for the x_net
        layers = tf.keras.layers
        input_features = layers.Input(shape=data_descriptions.sample_shape["input"], dtype=tf.int32)
        inner = layers.Embedding(self.num_class, self.hparams.d_model)(input_features)
        inner = ScaledPositionalEncoding(self.hparams.d_model)(inner)
        inner = layers.Dropout(self.hparams.rate)(inner)  # self.hparams.rate
        self.x_net = tf.keras.Model(inputs=input_features, outputs=inner, name="x_net")
        print(self.x_net.summary())

        # y_net for target
        input_labels = layers.Input(shape=[None, self.feat_dim * self.reduction_factor],
                                    dtype=tf.float32)
        inner = self.prenet(input_labels)
        inner = layers.Dense(self.hparams.d_model)(inner)
        inner = ScaledPositionalEncoding(self.hparams.d_model)(inner)
        inner = layers.Dropout(self.hparams.rate)(inner)
        self.y_net = tf.keras.Model(inputs=input_labels, outputs=inner, name="y_net")
        print(self.y_net.summary())

        # transformer layer
        self.transformer = Transformer(
            self.hparams.d_model,
            self.hparams.num_heads,
            self.hparams.num_encoder_layers,
            self.hparams.num_decoder_layers,
            self.hparams.dff,
            self.hparams.rate
        )

    def call(self, samples, training: bool = None):
        x0 = samples["input"]
        x = self.x_net(x0, training=training)
        y0 = samples['output']
        ori_lens = tf.shape(y0)[1]
        reduction_output_length = samples['output_length']
        if self.reduction_factor > 1:
            y0 = self._pad_and_reshape(y0, ori_lens)
            reduction_output_length = (samples['output_length'] - 1) // self.reduction_factor + 1
        y0 = self.initialize_input_y(y0)
        y = self.y_net(y0, training=training)
        output_mask, input_mask = create_multihead_mask(y0, reduction_output_length, x0, reverse=True)
        y, attention_weights = self.transformer(
            x,
            y,
            input_mask,
            output_mask,
            input_mask,
            training=training,
            return_attention_weights=True,
        )
        before_outs = self.feat_out(y)
        logits_stack = self.prob_out(y)
        logits_stack = self._pad_and_reshape(logits_stack, ori_lens, reverse=True)
        before_outs = self._pad_and_reshape(before_outs, ori_lens, reverse=True)
        if self.hparams.clip_outputs:
            maximum = - self.hparams.clip_max_value - self.hparams.clip_lower_bound_decay
            maximum = tf.maximum(before_outs, maximum)
            before_outs = tf.minimum(maximum, self.hparams.clip_max_value)
        # after_outs, shape: [batch, y_steps, feat_dim]
        after_outs = before_outs + self.postnet(before_outs, training=training)
        if self.hparams.clip_outputs:
            maximum = - self.hparams.clip_max_value - self.hparams.clip_lower_bound_decay
            maximum = tf.maximum(after_outs, maximum)
            after_outs = tf.minimum(maximum, self.hparams.clip_max_value)
        return before_outs, after_outs, logits_stack, attention_weights

    def time_propagate(self, encoder_output, memory_mask, outs, step):
        """
        Synthesize one step frames

        Args:
            encoder_output: the encoder output, shape: [batch, x_steps, eunits]
            memory_mask: the encoder output mask, shape: [batch, 1, 1, x_steps]
            outs (TensorArray): previous outputs
            step: the current step number
        Returns::

            out: new frame outpus, shape: [batch, feat_dim * reduction_factor]
            logit: new stop token prediction logit, shape: [batch, reduction_factor]
            attention_weights (list): the corresponding attention weights,
                each element in the list represents the attention weights of one decoder layer
                shape: [batch, num_heads, seq_len_q, seq_len_k]
        """

        prev_outs = tf.transpose(outs.stack(), [1, 0, 2])
        output_mask = generate_square_subsequent_mask(step)
        # propagate 1 step
        y = self.y_net(prev_outs, training=False)
        decoder_out, attention_weights = self.transformer.decoder(
            y,
            encoder_output,
            tgt_mask=output_mask,
            memory_mask=memory_mask,
            return_attention_weights=True,
            training=False,
        )
        decoder_out = decoder_out[:, -1, :] # [batch, feat_dim]
        out = self.feat_out(decoder_out)
        logit = self.prob_out(decoder_out)
        return out, logit, attention_weights

    def synthesize(self, samples):
        """
        Synthesize acoustic features from the input texts

        Args:
            samples: the data source to be synthesized
        Returns::

            after_outs: the corresponding synthesized acoustic features
            attn_weights_stack: the corresponding attention weights
        """

        x0 = samples["input"]
        input_length = samples['input_length']
        _, input_mask = create_multihead_mask(None, None, x0, reverse=True)
        x = self.x_net(x0, training=False)
        encoder_output = self.transformer.encoder(x, input_mask, training=False)

        batch = tf.shape(x0)[0]
        initial_y = tf.zeros([batch, self.feat_dim * self.reduction_factor])

        outs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        outs = outs.write(0, initial_y)
        logits = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        attn_weights = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        y_index = 0
        max_output_len = self.hparams.max_output_length * input_length[0] // self.reduction_factor
        for _ in tf.range(max_output_len):
            y_index += 1
            out, logit, new_weight = self.time_propagate(encoder_output, input_mask, outs, y_index)
            outs = outs.write(y_index, out)
            logits = logits.write(y_index-1, logit)
            attn_weights = attn_weights.write(y_index-1, new_weight[-1][:, :, -1])
            probs = tf.nn.sigmoid(logit)
            time_to_end = probs > self.hparams.end_prob
            time_to_end = tf.reduce_any(time_to_end)
            if time_to_end:
                break

        logits_stack = tf.transpose(logits.stack(), [1, 0, 2]) # [batch, y_steps, reduction_factor]
        # before_outs: [batch, y_steps, feat_dim*reduction_factor]
        before_outs = tf.transpose(outs.stack(), [1, 0, 2])[:, 1:]
        # attn_weights_stack: [batch, head, y_steps, x_steps]
        attn_weights_stack = tf.transpose(attn_weights.stack(), [1, 2, 0, 3])
        after_outs = self._synthesize_post_net(before_outs, logits_stack)
        return after_outs, attn_weights_stack
