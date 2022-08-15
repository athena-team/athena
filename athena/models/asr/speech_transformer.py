# coding=utf-8
# Copyright (C) ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Xiaoning Lei
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

from typing import List

from absl import logging
import tensorflow as tf
from athena.models.base import BaseModel
from athena.models.lm.kenlm import NgramLM
from athena.loss import Seq2SeqSparseCategoricalCrossentropy
from athena.metrics import Seq2SeqSparseCategoricalAccuracy
from athena.utils.misc import generate_square_subsequent_mask, insert_sos_in_labels, create_multihead_mask, \
    mask_finished_preds, mask_finished_scores
from athena.layers.commons import PositionalEncoding
from athena.layers.transformer import Transformer
from athena.utils.hparam import register_and_parse_hparams
# from ..tools.ctc_scorer import CTCPrefixScoreTH
from athena.tools.ctc_decoder import ctc_prefix_beam_decoder


class SpeechTransformer(BaseModel):
    """ Standard implementation of a SpeechTransformer. Model mainly consists of three parts:
    the x_net for input preparation, the y_net for output preparation and the transformer itself
    """
    default_config = {
        "return_encoder_output": False,
        "num_filters": 512,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 12,
        "num_decoder_layers": 6,
        "dff": 1280,
        "max_position": 800,
        "dropout_rate": 0.1,
        "schedual_sampling_rate": 0.9,
        "label_smoothing_rate": 0.0,
        "unidirectional": False,
        "look_ahead": 0,
        "conv_module_kernel_size": 0
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        self.num_class = data_descriptions.num_class + 1
        self.sos = self.num_class - 1
        self.eos = self.num_class - 1
        self.ignore_pad_value = 0
        ls_rate = self.hparams.label_smoothing_rate
        self.loss_function = Seq2SeqSparseCategoricalCrossentropy(
            num_classes=self.num_class, eos=self.eos, label_smoothing=ls_rate
        )
        self.metric = Seq2SeqSparseCategoricalAccuracy(eos=self.eos, name="Accuracy")

        # for the x_net
        num_filters = self.hparams.num_filters
        d_model = self.hparams.d_model
        layers = tf.keras.layers
        input_features = layers.Input(shape=data_descriptions.sample_shape["input"], dtype=tf.float32)
        inner = layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(input_features)
        inner = layers.BatchNormalization()(inner)
        inner = tf.nn.relu6(inner)
        inner = layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(inner)
        inner = layers.BatchNormalization()(inner)

        inner = tf.nn.relu6(inner)
        _, _, dim, channels = inner.get_shape().as_list()
        output_dim = dim * channels
        inner = layers.Reshape((-1, output_dim))(inner)

        inner = layers.Dense(d_model, activation=tf.nn.relu6)(inner)
        inner = PositionalEncoding(d_model, max_position=self.hparams.max_position, scale=False)(inner)
        inner = layers.Dropout(self.hparams.dropout_rate)(inner)
        self.x_net = tf.keras.Model(inputs=input_features, outputs=inner, name="x_net")
        print(self.x_net.summary())

        # y_net for target
        input_labels = layers.Input(shape=data_descriptions.sample_shape["output"], dtype=tf.int32)
        inner = layers.Embedding(self.num_class, d_model)(input_labels)
        inner = PositionalEncoding(d_model, max_position=self.hparams.max_position, scale=True)(inner)
        inner = layers.Dropout(self.hparams.dropout_rate)(inner)
        self.y_net = tf.keras.Model(inputs=input_labels, outputs=inner, name="y_net")#TODO:need change
        print(self.y_net.summary())

        # transformer layer
        self.transformer = Transformer(
            self.hparams.d_model,
            self.hparams.num_heads,
            self.hparams.num_encoder_layers,
            self.hparams.num_decoder_layers,
            self.hparams.dff,
            self.hparams.dropout_rate,
            unidirectional=self.hparams.unidirectional,
            look_ahead=self.hparams.look_ahead,
            conv_module_kernel_size=self.hparams.conv_module_kernel_size
        )

        # last layer for output
        self.final_layer = layers.Dense(self.num_class, input_shape=(d_model,))

        # some temp function
        self.random_num = tf.random_uniform_initializer(0, 1)

    def call(self, samples, training: bool = None):
        x0 = samples["input"]
        y0 = insert_sos_in_labels(samples["output"], self.sos)
        x = self.x_net(x0, training=training)
        y = self.y_net(y0, training=training)
        input_length = self.compute_logit_length(samples["input_length"])
        input_mask, output_mask = create_multihead_mask(x, input_length, y0)
        y, encoder_output = self.transformer(
            x,
            y,
            input_mask,
            output_mask,
            input_mask,
            training=training,
            return_encoder_output=True,
        )
        y = self.final_layer(y)
        if self.hparams.return_encoder_output:
            return y, encoder_output
        return y

    def compute_logit_length(self, input_length):
        """ used for get logit length """
        input_length = tf.cast(input_length, tf.float32)
        logit_length = tf.math.ceil(input_length / 2)
        logit_length = tf.math.ceil(logit_length / 2)
        logit_length = tf.cast(logit_length, tf.int32)
        return logit_length

    def _forward_encoder(self, speech, speech_length, training: bool = None):
        x = self.x_net(speech, training=training)
        input_length = self.compute_logit_length(speech_length)
        input_mask, _ = create_multihead_mask(x, input_length, None)
        # 1. Encoder
        encoder_output = self.transformer.encoder(x, input_mask, training=training)  # (B, maxlen, encoder_dim)
        return encoder_output, input_mask
    
    def _forward_encoder_log_ctc(self, samples, final_layer, training: bool = None):
        speech = samples['input']
        speech_length = samples['input_length']

        x = self.x_net(speech, training=training)
        input_length = self.compute_logit_length(speech_length)
        input_mask, _ = create_multihead_mask(x, input_length, None)
        encoder_output = self.transformer.encoder(x, input_mask, training=training)  # (B, maxlen, encoder_dim)
        predictions = final_layer(encoder_output, training=training)
        predictions = tf.nn.log_softmax(predictions, axis=-1)
        return predictions, input_mask
    
    def ctc_prefix_beam_search(
            self, samples, hparams, ctc_final_layer
    ) -> List[int]:
        speech = samples["input"]
        speech_lengths = samples["input_length"]
        beam_size, ctc_weight = hparams.beam_size, hparams.ctc_weight
        assert speech.shape[0] == speech_lengths.shape[0]

        encoder_out, encoder_mask = self._forward_encoder(speech, speech_lengths, training=False)

        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out = ctc_prefix_beam_decoder(
            encoder_out, ctc_final_layer, beam_size, blank_id=self.sos)  # tf.nn.ctc_beam_search_decoder

        return tf.convert_to_tensor(hyps[0][0])[tf.newaxis, :]

    def attention_rescoring(
            self,
            samples,
            hparams,
            ctc_final_layer: tf.keras.layers.Dense,
            lm_model: BaseModel = None
    ) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out

        Args:
            samples :
            hparams : inference_config
            ctc_final_layer : encoder final dense layer to output ctc prob.
            lm_model :

        Returns:
            List[int]: Attention rescoring result
        """
        speech = samples["input"]
        speech_lengths = samples["input_length"]
        beam_size, ctc_weight = hparams.beam_size, hparams.ctc_weight
        assert speech.shape[0] == speech_lengths.shape[0]

        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size

        encoder_out, encoder_mask = self._forward_encoder(speech, speech_lengths, training=False)  # (B, maxlen, encoder_dim)
        hyps, encoder_out = ctc_prefix_beam_decoder(
            encoder_out, ctc_final_layer, beam_size, blank_id=self.sos)

        assert len(hyps) == beam_size

        sequence = [tf.convert_to_tensor(hyp[0]) for hyp in hyps]
        hyps_pad = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, padding='post', value=self.ignore_pad_value, dtype='int32'
        )

        hyps_pad_len = [len(i)+1 for i in sequence]
        hyps_pad = insert_sos_in_labels(hyps_pad, self.sos)
        output_mask, _ = create_multihead_mask(hyps_pad, hyps_pad_len, None, reverse=True)

        encoder_out = tf.tile(encoder_out, [beam_size, 1, 1])  # encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = tf.zeros([beam_size, 1, 1, encoder_out.shape[1]], dtype=tf.dtypes.float32)

        hyps_emb = self.y_net(hyps_pad, training=False)

        y_mask = output_mask

        decoder_out = self.transformer.decoder(
            hyps_emb,  # [20, 25, 256]
            encoder_out,  # [20, 46, 256]
            y_mask,  # [20,1, 1, 25]
            encoder_mask,  # [20, 1, 1, 46]
            training=False,
        )

        decoder_out = self.final_layer(decoder_out)

        decoder_out = tf.nn.log_softmax(decoder_out, axis=-1)
        decoder_out = decoder_out.numpy()

        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0

        lm_weight = hparams.lm_weight if lm_model is not None else 0.0
        at_weight = hparams.at_weight
        if lm_weight > 0.0:
            if isinstance(lm_model, NgramLM):
                lm_logprob = lm_model.forward(hyps_pad, hyps_pad_len, training=False,
                                              ignored_id=[self.eos, self.sos, self.ignore_pad_value])
            else:
                lm_logist = lm_model.forward(hyps_pad, hyps_pad_len, training=False)
                lm_logprob = tf.nn.log_softmax(lm_logist)

        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w] * at_weight
                if lm_weight > 0.0:
                    if isinstance(lm_model, NgramLM):
                        score += lm_logprob[i][j] * lm_weight
                    else:
                        score += lm_logprob[i][j][w] * lm_weight
            score += decoder_out[i][len(hyp[0])][self.eos] * at_weight
            if lm_weight > 0.0:
                if isinstance(lm_model, NgramLM):
                    score += lm_logprob[i][len(hyp[0])] * lm_weight
                else:
                    score += lm_logprob[i][len(hyp[0])][self.eos] * lm_weight

            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        return tf.convert_to_tensor(hyps[best_index][0])[tf.newaxis, :]  # hyps[best_index][0],

    def freeze_beam_search(self, samples, beam_size=1):
        """ beam search for freeze only support batch=1

        Args:
            samples: the data source to be decoded
            beam_size: beam size
        """
        x0 = samples["input"]
        batch_size = tf.shape(x0)[0]
        x = self.x_net(x0, training=False)
        input_length = self.compute_logit_length(samples["input_length"])
        input_mask, _ = create_multihead_mask(x, input_length, None)
        # 1. Encoder
        encoder_output = self.transformer.encoder(x, input_mask, training=False)# (B, maxlen, encoder_dim)
        maxlen = tf.shape(encoder_output)[1]
        encoder_dim = tf.shape(encoder_output)[2]
        running_size = beam_size

        # repeat for beam_size
        input_mask = tf.tile(input_mask, [1, beam_size, 1, 1])
        input_mask = tf.transpose(input_mask, [1, 0, 2, 3])

        encoder_output = tf.tile(encoder_output, [1, beam_size, 1])
        encoder_output = tf.transpose(encoder_output, [1, 0, 2])

        hyps = tf.ones([running_size, 1], dtype=tf.int32) * self.sos # (B*N, max_len)
        scores = tf.constant([[0.0]+[-float('inf')] * (beam_size-1)], shape=(beam_size, 1), dtype=tf.float32)
        scores = tf.tile(scores, [batch_size, 1])

        end_flag = tf.zeros_like(scores, dtype=tf.int32)
        # 2. Decoder forward step by step
        for step in tf.range(1, maxlen+1):
            tf.autograph.experimental.set_loop_options(
                    shape_invariants=[
                        (hyps, tf.TensorShape([None, None])),
                        (scores, tf.TensorShape([None, None])),
                        (end_flag, tf.TensorShape([None,None]))
                        ])
            if tf.reduce_sum(end_flag) == running_size:
                break
            # 2.1 Forward decoder step
            output_mask = generate_square_subsequent_mask(step)

            y0 = self.y_net(hyps, training=False)

            decoder_outputs = self.transformer.decoder(
                y0,
                encoder_output,
                tgt_mask=output_mask,
                memory_mask=input_mask,
                training=False,
            )
            logits = self.final_layer(decoder_outputs)
            logit = logits[:, -1, :]
            logprob = tf.math.log(tf.nn.softmax(logit))


            # 2.3 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = tf.math.top_k(logprob, k=beam_size)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.4 Seconde beam prune: select topk score with history
            scores = scores + top_k_logp
            scores = tf.reshape(scores, [batch_size, beam_size*beam_size])

            scores, best_k_index = tf.math.top_k(scores, k=beam_size)
            scores = tf.reshape(scores, [batch_size*beam_size, 1])
            # 2.5. Compute base index in top_k_index
            row_index = best_k_index // beam_size
            col_index = best_k_index % beam_size

            batch_index = tf.range(batch_size)
            batch_index = tf.reshape(batch_index, [batch_size, 1])
            batch_index = tf.tile(batch_index, [1, beam_size])
            batch_index = tf.reshape(batch_index, [batch_size, beam_size, 1])

            row_index = tf.expand_dims(row_index, axis=2)
            col_index = tf.expand_dims(col_index, axis=2)
            indices = tf.concat([batch_index, row_index, col_index], axis=2)
            top_k_index = tf.reshape(top_k_index, [batch_size, beam_size, beam_size])
            # 2.6 Update best hyps
            best_k_pred = tf.gather_nd(top_k_index, indices)
            best_k_pred = tf.reshape(best_k_pred, [batch_size*beam_size, 1])

            last_best_k_index = batch_index * beam_size + row_index
            last_best_k_index = tf.reshape(last_best_k_index, [running_size, 1])
            last_best_k_hyps = tf.gather(hyps, last_best_k_index,axis=0)
            last_best_k_hyps = tf.reshape(last_best_k_hyps, [running_size, -1])
            hyps = tf.concat([last_best_k_hyps, best_k_pred], axis=1)
            # 2.7 Update end flag
            end_flag = tf.equal(hyps[:,-1], self.eos)
            end_flag = tf.cast(end_flag, dtype=tf.int32)
            end_flag = tf.expand_dims(end_flag, axis=1)

        batch_index = tf.range(batch_size)
        batch_index = tf.reshape(batch_index, [batch_size, 1])

        scores = tf.reshape(scores, [batch_size, beam_size])
        # 3. Select best of best
        best_index = tf.argmax(scores, axis=1)
        best_index = tf.reshape(best_index, [batch_size, 1])
        best_index = tf.cast(best_index, dtype=tf.int32)
        best_index = tf.concat([batch_index, best_index], axis=1)

        hyps = tf.reshape(hyps, [batch_size, beam_size, -1])
        best_hyps = tf.gather_nd(hyps, best_index)
        return best_hyps


    def beam_search(self, samples, hparams, lm_model=None):
        """ batch beam search for transformer model

        Args:
            samples: the data source to be decoded
            beam_size: beam size
            lm_model: rnnlm that used for beam search

        """
        x0 = samples["input"]
        batch_size = tf.shape(x0)[0]
        beam_size = hparams.beam_size
        x = self.x_net(x0, training=False)
        input_length = self.compute_logit_length(samples["input_length"])
        input_mask, _ = create_multihead_mask(x, input_length, None)
        # 1. Encoder
        encoder_output = self.transformer.encoder(x, input_mask, training=False)# (B, maxlen, encoder_dim)
        maxlen = tf.shape(encoder_output)[1]
        encoder_dim = tf.shape(encoder_output)[2]
        running_size = batch_size * beam_size

        # repeat for beam_size
        a, b, c, d = input_mask.get_shape().as_list()
        input_mask = tf.tile(input_mask, [1, beam_size, 1, 1])
        input_mask = tf.reshape(input_mask, [a*beam_size, b, c, d])

        a, b, c = encoder_output.get_shape().as_list()
        encoder_output = tf.tile(encoder_output, [1, beam_size, 1])
        encoder_output = tf.reshape(encoder_output, [a * beam_size, b, c])

        hyps = tf.ones([running_size, 1], dtype=tf.int32) * self.sos # (B*N, max_len)
        scores = tf.constant([[0.0] + [-float('inf')] * (beam_size - 1)], shape=(beam_size, 1), dtype=tf.float32)
        scores = tf.tile(scores, [batch_size, 1])

        end_flag = tf.zeros_like(scores, dtype=tf.int32)
        # 2. Decoder forward step by step
        ctc_states = None
        for step in tf.range(1, maxlen + 1):
            if tf.reduce_sum(end_flag) == running_size:
                break
            # 2.1 Forward decoder step
            output_mask = generate_square_subsequent_mask(step)

            y0 = self.y_net(hyps, training=False)

            decoder_outputs = self.transformer.decoder(
                y0,
                encoder_output,
                tgt_mask=output_mask,
                memory_mask=input_mask,
                training=False,
            )

            logits = self.final_layer(decoder_outputs)
            logit = logits[:, -1, :]
            logprob = tf.math.log(tf.nn.softmax(logit))
            # 2.2 add encoder ctc score
            # if hparams.ctc_weight != 0:
            #     ctc_logits = ctc_decoder(encoder_output, training=False)
            #     ctc_logits = tf.math.log(tf.nn.softmax(ctc_logits))
            #     xlen = tf.fill(1, ctc_logits.shape[1])
            #     self.impl = CTCPrefixScoreTH(ctc_logits, xlen, 0, self.eos)
            #     pre_top_k_logp, pre_top_k_index = tf.math.top_k(logprob, k=hparams.pre_beam_size)
            #     top_k_logp, top_k_index = tf.math.top_k(logprob, k=beam_size)
            #     ctc_local_scores, ctc_states = self.impl(top_k_index, ctc_states, pre_top_k_index)
                # TODO 打分合并

            # 2.3 add language model score
            if lm_model is not None:
                hyps_len = tf.cast(hyps != self.eos, dtype=tf.int32)
                if self.sos != self.eos:
                    hyps_len = hyps_len - 1
                hyps_len = tf.reduce_sum(hyps_len, axis=-1)
                lm_logits = lm_model.forward(hyps, hyps_len, training=False)
                lm_logit = lm_logits[:, -1, :]
                lm_logprob = tf.math.log(tf.nn.softmax(lm_logit))
                lm_weight = hparams.lm_weight
                logprob = ((1 - lm_weight) * logprob) + (lm_weight * lm_logprob)
            # 2.4 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = tf.math.top_k(logprob, k=beam_size)
            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos)
            # 2.5 Seconde beam prune: select topk score with history
            scores = scores + top_k_logp
            scores = tf.reshape(scores, [batch_size, beam_size * beam_size])

            scores, best_k_index = tf.math.top_k(scores, k=beam_size)
            scores = tf.reshape(scores, [batch_size * beam_size, 1])
            # 2.6. Compute base index in top_k_index
            row_index = best_k_index // beam_size
            col_index = best_k_index % beam_size

            batch_index = tf.range(batch_size)
            batch_index = tf.reshape(batch_index, [batch_size, 1])
            batch_index = tf.tile(batch_index, [1, beam_size])
            batch_index = tf.reshape(batch_index, [batch_size, beam_size, 1])

            row_index = tf.expand_dims(row_index, axis=2)
            col_index = tf.expand_dims(col_index, axis=2)
            indices = tf.concat([batch_index, row_index, col_index], axis=2)
            top_k_index = tf.reshape(top_k_index, [batch_size, beam_size, beam_size])
            # 2.7 Update best hyps
            best_k_pred = tf.gather_nd(top_k_index, indices)
            best_k_pred = tf.reshape(best_k_pred, [batch_size * beam_size, 1])

            last_best_k_index = batch_index * beam_size + row_index
            last_best_k_index = tf.reshape(last_best_k_index, [running_size, 1])
            last_best_k_hyps = tf.gather(hyps, last_best_k_index, axis=0)
            last_best_k_hyps = tf.reshape(last_best_k_hyps, [running_size, -1])
            hyps = tf.concat([last_best_k_hyps, best_k_pred], axis=1)
            # 2.8 Update end flag
            end_flag = tf.equal(hyps[:, -1], self.eos)
            end_flag = tf.cast(end_flag, dtype=tf.int32)
            end_flag = tf.expand_dims(end_flag, axis=1)

        batch_index = tf.range(batch_size)
        batch_index = tf.reshape(batch_index, [batch_size, 1])

        scores = tf.reshape(scores, [batch_size, beam_size])
        # 3. Select best of best
        best_index = tf.argmax(scores, axis=1)
        best_index = tf.reshape(best_index, [batch_size, 1])
        best_index = tf.cast(best_index, dtype=tf.int32)
        best_index = tf.concat([batch_index, best_index], axis=1)

        hyps = tf.reshape(hyps, [batch_size, beam_size, -1])
        best_hyps = tf.gather_nd(hyps, best_index)
        return best_hyps

    def freeze_ctc_prefix_beam_search(
            self, samples, ctc_final_layer, hparams=None,beam_size=1
    ) -> List[int]:
        speech = samples["input"]
        speech_lengths = samples["input_length"]
        x = self.x_net(speech, training=False)
        input_length = self.compute_logit_length(speech_lengths)
        input_mask, _ = create_multihead_mask(x, input_length, None)
        # 1. Encoder
        encoder_output = self.transformer.encoder(x, input_mask, training=False)

        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        # hyps, encoder_out = ctc_prefix_beam_decoder(
        #     encoder_out, ctc_final_layer, beam_size, blank_id=self.sos)  # tf.nn.ctc_beam_search_decoder
        # return tf.convert_to_tensor(hyps[0][0])[tf.newaxis, :]

        ctc_probs = tf.nn.log_softmax(ctc_final_layer(encoder_output), axis=2)  # (1, maxlen, vocab_size)
        ctc_probs = tf.transpose(ctc_probs, (1, 0, 2))
        decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(  # (max_time, batch_size, num_classes)
            ctc_probs, input_length, beam_width=beam_size, top_paths=beam_size
        )
        return decoded[0].values[tf.newaxis, :]

    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        if model_type == "":
            return
        if model_type == "mpc":
            logging.info("loading from pretrained mpc model")
            self.x_net = pretrained_model.x_net
            self.transformer.encoder = pretrained_model.encoder
        elif model_type == "SpeechTransformer":
            logging.info("loading from pretrained SpeechTransformer model")
            self.x_net = pretrained_model.x_net
            self.y_net = pretrained_model.y_net
            self.transformer = pretrained_model.transformer
            self.final_layer = pretrained_model.final_layer
        else:
            raise ValueError("NOT SUPPORTED")



