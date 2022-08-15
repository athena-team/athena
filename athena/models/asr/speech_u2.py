# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Chaoyang Mei; Jianwei Sun
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

"""This code is modified from https://github.com/wenet-e2e/wenet.git."""

from typing import List, Tuple

from absl import logging

from athena.layers.u2.conformer_u2 import ConformerU2
from athena.layers.u2.embedding import PositionalEncodingU2
from athena.layers.u2.subsampling import Conv2dSubsampling4
from athena.layers.u2.transformer_u2 import TransformerU2
import tensorflow as tf

from athena.models.base import BaseModel
from athena.loss import Seq2SeqSparseCategoricalCrossentropy
from athena.metrics import Seq2SeqSparseCategoricalAccuracy
from athena.utils.misc import generate_square_subsequent_mask_u2, insert_sos_in_labels, create_multihead_mask_u2, \
    mask_finished_preds, mask_finished_scores, add_optional_chunk_mask_u2
from athena.layers.commons import PositionalEncoding
from athena.utils.hparam import register_and_parse_hparams
from athena.tools.ctc_decoder import ctc_prefix_beam_decoder


class SpeechU2(BaseModel):
    """ Base model for U2
    """
    default_config = {
        "return_encoder_output": False,
        "max_position": 5000,
        "use_dynamic_chunk": True,   # whether use dynamic encoder chunk for training
        "use_dynamic_left_chunk": False,    # you can choose dynamic encoder left chunk for training when use dynamic chunk
        "static_chunk_size": -1,    # if static_chunk_size > 0, encoder chunk size will be static_chunk_size, when dynamic chunk is turn off
        "schedual_sampling_rate": 0.9,
        "label_smoothing_rate": 0.0,
        "unidirectional": False,
        "look_ahead": 0,
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        self.num_class = data_descriptions.num_class + 1
        self.sos = self.num_class - 1
        self.eos = self.num_class - 1
        self.feats_dim = data_descriptions.audio_featurizer.dim
        ls_rate = self.hparams.label_smoothing_rate
        self.loss_function = Seq2SeqSparseCategoricalCrossentropy(
            num_classes=self.num_class, eos=self.eos, label_smoothing=ls_rate
        )
        self.metric = Seq2SeqSparseCategoricalAccuracy(eos=self.eos, name="Accuracy")

        self.use_dynamic_chunk = self.hparams.use_dynamic_chunk
        self.use_dynamic_left_chunk = self.hparams.use_dynamic_left_chunk
        self.static_chunk_size = self.hparams.static_chunk_size

        # some temp function
        self.random_num = tf.random_uniform_initializer(0, 1)

        self.tf_funtion_enabled = False

    def enable_tf_funtion(self):
        if not self.tf_funtion_enabled:
            self._encoder_forward_chunk = tf.function(self._encoder_forward_chunk,
                                                      input_signature=[
                                                         tf.TensorSpec(shape=[1, None, self.feats_dim, 1],
                                                                       dtype=tf.float32),
                                                         tf.TensorSpec(shape=None, dtype=tf.int32),
                                                         tf.TensorSpec(shape=None, dtype=tf.int32),
                                                         tf.TensorSpec(shape=[1, None, self.hparams.d_model],
                                                                       dtype=tf.float32),
                                                         tf.TensorSpec(
                                                             shape=[self.hparams.num_encoder_layers, 1, None,
                                                                    self.hparams.d_model], dtype=tf.float32),
                                                         tf.TensorSpec(
                                                             shape=[self.hparams.num_encoder_layers, 1, None,
                                                                    self.hparams.d_model], dtype=tf.float32)])
            self._forward_transformer_encoder = tf.function(self._forward_transformer_encoder,
                                                            input_signature=[
                                                                tf.TensorSpec(shape=[None, None, self.hparams.d_model],
                                                                              dtype=tf.float32),
                                                                tf.TensorSpec(shape=[None, 1, None, None],
                                                                              dtype=tf.bool),
                                                                tf.TensorSpec(shape=None, dtype=tf.bool), ]
                                                            )
            self._forward_decoder = tf.function(self._forward_decoder,
                                                input_signature=[tf.TensorSpec(shape=[None, None, self.hparams.d_model], dtype=tf.float32),
                                                                 tf.TensorSpec(shape=[None, 1, 1, None], dtype=tf.bool),
                                                                 tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                                                                 tf.TensorSpec(shape=[None, 1, None, None], dtype=tf.bool)
                                                                 ])

            self.tf_funtion_enabled = True
        pass

    def call(self, samples, training: bool = None):
        x0 = samples["input"]
        y0 = insert_sos_in_labels(samples["output"], self.sos)
        x0_length = samples["input_length"]

        x0_mask_pad, output_mask = create_multihead_mask_u2(x0, x0_length, y0)
        x, pos_emb, x_mask = self.x_net(x0, x0_mask_pad, training=training)
        x_length = self.x_net.compute_logit_length(x0_length)  # tf.shape(x)[-2]
        y = self.y_net(y0, training=training)

        input_chunk_masks = add_optional_chunk_mask_u2(x, x_mask, tf.reduce_max(x_length),
                                                       use_dynamic_chunk=self.use_dynamic_chunk,
                                                       use_dynamic_left_chunk=self.use_dynamic_left_chunk,
                                                       decoding_chunk_size=0,
                                                       static_chunk_size=self.static_chunk_size,
                                                       num_decoding_left_chunks=-1)

        y, encoder_output = self.transformer(
            x,
            y,
            src_mask=input_chunk_masks,
            tgt_mask=output_mask,
            memory_mask=x_mask,
            training=training,
            return_encoder_output=True,
        )
        y = self.final_layer(y)
        if self.hparams.return_encoder_output:
            return y, encoder_output
        return y

    def compute_logit_length(self, input_length):
        """ used for get logit length """
        return self.x_net.compute_logit_length(input_length)

    # following functions are used to decoding.

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 256], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[None, 1, None, None], dtype=tf.bool),
    #                               tf.TensorSpec(shape=None, dtype=tf.bool), ])
    def _forward_transformer_encoder(self, x, x_mask, training=None):
        encoder_out, encoder_mask = self.transformer.encoder(x, x_mask,
                                                             training=training)  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    def _forward_encoder(self,
                         speech: tf.Tensor,
                         speech_length: tf.Tensor,
                         decoding_chunk_size: int = -1,
                         num_decoding_left_chunks: int = -1,
                         simulate_streaming: bool = False,
                         use_dynamic_left_chunk: bool = False,
                         static_chunk_size: int = -1,
                         training: bool = None
                         ) -> Tuple[tf.Tensor, tf.Tensor]:
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self._encoder_forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            speech_mask, _ = create_multihead_mask_u2(speech, speech_length, None)
            x, pos_emb, x_mask = self.x_net(speech, speech_mask, training=training)
            x_length = self.x_net.compute_logit_length(speech_length)
            input_mask = add_optional_chunk_mask_u2(x, x_mask, tf.reduce_max(x_length), use_dynamic_chunk=True,
                                                    use_dynamic_left_chunk=use_dynamic_left_chunk,
                                                    decoding_chunk_size=decoding_chunk_size,
                                                    static_chunk_size=static_chunk_size,
                                                    num_decoding_left_chunks=num_decoding_left_chunks)
            encoder_out, encoder_mask = self._forward_transformer_encoder(x, input_mask,
                                                                          training=training)  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    # @tf.function(input_signature=[tf.TensorSpec(shape=[1, None, 80, 1], dtype=tf.float32),
    #                               tf.TensorSpec(shape=None, dtype=tf.int32),
    #                               tf.TensorSpec(shape=None, dtype=tf.int32),
    #                               tf.TensorSpec(shape=[1, None, 256], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[12, 1, None, 256], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[12, 1, None, 256], dtype=tf.float32)])
    def _encoder_forward_chunk(
            self,
            xs: tf.Tensor,
            offset: int,
            required_cache_size: int,
            subsampling_cache: tf.Tensor,
            elayers_output_cache: tf.Tensor,
            conformer_cnn_cache: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor],
               List[tf.Tensor]]:
        """ Forward just one chunk

        Args:
            xs (tf.Tensor), shape=[B(1), T, feat_dim, 1]: chunk input
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            subsampling_cache, shape=[B(1), T, feat_dim]: subsampling cache
            elayers_output_cache, shape=[num_layers, 1, T, d_model]:
                transformer/conformer encoder layers output cache
            conformer_cnn_cache, shape=[num_layers, 1, T, d_model]: conformer
                cnn cache

        Returns:
            tf.Tensor: output of current input xs
            tf.Tensor: subsampling cache required for next chunk computation
            tf.Tensor: encoder layers output cache required for next
                chunk computation
            tf.Tensor: conformer cnn cache

        """
        # assert tf.shape(xs)[0] == tf.constant(1)
        # tmp_masks is just for interface compatibility
        tmp_masks = tf.zeros((1, 1, 1, tf.shape(xs)[1]), dtype=tf.bool)
        xs, pos_emb, x_mask = self.x_net(xs, tmp_masks, offset)
        if tf.shape(subsampling_cache)[1] > 0:
            cache_size = tf.shape(subsampling_cache)[1]
            xs = tf.concat((subsampling_cache, xs), axis=1)
        else:
            cache_size = 0
        # pos_emb = self.embed.position_encoding(offset - cache_size, xs.shape[1])
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = tf.shape(xs)[1]
        else:
            next_cache_start = tf.math.maximum(tf.shape(xs)[1] - required_cache_size,
                                               0)  # max(tf.shape(xs)[1] - required_cache_size, 0)
        r_subsampling_cache = xs[:, next_cache_start:, :]
        # Real mask for transformer/conformer layers
        masks = tf.zeros((1, tf.shape(xs)[1]), dtype=tf.bool)
        masks = tf.expand_dims(masks, 1)
        r_elayers_output_cache = []
        r_conformer_cnn_cache = []
        for i, layer in enumerate(self.transformer.encoder.layers):
            attn_cache = elayers_output_cache[i]
            cnn_cache = conformer_cnn_cache[i]
            xs, _, new_cnn_cache = layer(xs,  # @tf.function WARNING
                                         masks,
                                         output_cache=attn_cache,
                                         cnn_cache=cnn_cache)
            r_elayers_output_cache.append(xs[:, next_cache_start:, :])
            r_conformer_cnn_cache.append(new_cnn_cache)

        r_elayers_output_cache = tf.stack(r_elayers_output_cache, 0)
        r_conformer_cnn_cache = tf.stack(r_conformer_cnn_cache, 0)
        r_offset = offset + tf.shape(xs)[1] - cache_size
        return (xs[:, cache_size:, :], r_offset, r_subsampling_cache,
                r_elayers_output_cache, r_conformer_cnn_cache)

    def _encoder_forward_chunk_by_chunk(
            self,
            speech: tf.Tensor,
            decoding_chunk_size: int,
            num_decoding_left_chunks: int = -1,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            speech (tf.Tensor): (1, max_len, feat_dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.transformer.encoder.static_chunk_size > 0 or self.transformer.encoder.use_dynamic_chunk
        subsampling_cache, elayers_output_cache, conformer_cnn_cache, right_context, subsampling = self.get_encoder_init_input()

        context = right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        num_frames = speech.shape[1]
        offset = 0
        outputs = []
        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = speech[:, cur:end, :]
            (y, offset, subsampling_cache, elayers_output_cache,
             conformer_cnn_cache) = self._encoder_forward_chunk(chunk_xs, offset,
                                                                required_cache_size,
                                                                subsampling_cache,
                                                                elayers_output_cache,
                                                                conformer_cnn_cache, )
            outputs.append(y)
            # offset += y.shape[1]
        ys = tf.concat(outputs, 1)
        masks = tf.zeros((1, ys.shape[1]), dtype=tf.bool)
        masks = tf.expand_dims(masks, axis=1)
        return ys, masks

    def get_encoder_init_input(self):
        subsampling = self.x_net.subsampling_rate
        right_context = self.x_net.right_context

        subsampling_cache = tf.zeros([1, 0, self.hparams.d_model], dtype=tf.float32)
        elayers_output_cache = tf.zeros([self.hparams.num_encoder_layers, 1, 0, self.hparams.d_model], dtype=tf.float32)
        conformer_cnn_cache = tf.zeros([self.hparams.num_encoder_layers, 1, 0, self.hparams.d_model], dtype=tf.float32)

        return subsampling_cache, elayers_output_cache, conformer_cnn_cache, right_context, subsampling

    def _forward_decoder(self, encoder_out, encoder_mask, hyps_pad, output_mask):
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

        return decoder_out

    def ctc_prefix_beam_search(
            self, samples, hparams, ctc_final_layer
    ) -> List[int]:
        speech = samples["input"]
        speech_lengths = samples["input_length"]
        beam_size, ctc_weight, simulate_streaming, decoding_chunk_size, num_decoding_left_chunks = \
            hparams.beam_size, hparams.ctc_weight, hparams.simulate_streaming, hparams.decoding_chunk_size, hparams.num_decoding_left_chunks
        assert speech.shape[0] == speech_lengths.shape[0]

        encoder_out, encoder_mask = self._forward_encoder(speech, speech_lengths, training=False,
                                                          simulate_streaming=simulate_streaming, decoding_chunk_size=decoding_chunk_size,
                                                          num_decoding_left_chunks=num_decoding_left_chunks)

        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        # hyps, encoder_out = ctc_prefix_beam_decoder(
        #     encoder_out, ctc_final_layer, beam_size, blank_id=self.sos)  # tf.nn.ctc_beam_search_decoder
        # return tf.convert_to_tensor(hyps[0][0])[tf.newaxis, :]

        ctc_probs = tf.nn.log_softmax(ctc_final_layer(encoder_out), axis=2)  # (1, maxlen, vocab_size)
        ctc_probs = tf.transpose(ctc_probs, (1, 0, 2))
        sequence_length = self.x_net.compute_logit_length(speech_lengths)
        decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(  # (max_time, batch_size, num_classes)
            ctc_probs, sequence_length, beam_width=beam_size, top_paths=beam_size
        )

        return decoded[0].values[tf.newaxis, :]

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
        beam_size, ctc_weight, simulate_streaming, decoding_chunk_size, num_decoding_left_chunks = \
            hparams.beam_size, hparams.ctc_weight, hparams.simulate_streaming, hparams.decoding_chunk_size, hparams.num_decoding_left_chunks
        assert speech.shape[0] == speech_lengths.shape[0]

        # For attention rescoring we only support batch_size=1
        assert speech.shape[0] == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size

        # encoder_out, encoder_mask = self._forward_encoder(speech, speech_lengths,
        #                                                   training=False)  # (B, maxlen, encoder_dim)
        encoder_out, encoder_mask = self._forward_encoder(speech, speech_lengths, training=False,
                                                          simulate_streaming=simulate_streaming, decoding_chunk_size=decoding_chunk_size,
                                                          num_decoding_left_chunks=num_decoding_left_chunks)
        hyps, encoder_out = ctc_prefix_beam_decoder(
            encoder_out, ctc_final_layer, beam_size, blank_id=self.sos)

        assert len(hyps) == beam_size

        sequence = [tf.convert_to_tensor(hyp[0]) for hyp in hyps]
        hyps_pad = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, padding='post', value=0, dtype='int32'
        )

        hyps_pad_len = [len(i) + 1 for i in sequence]
        hyps_pad = insert_sos_in_labels(hyps_pad, self.sos)
        output_mask, _ = create_multihead_mask_u2(hyps_pad, hyps_pad_len, None, reverse=True)

        encoder_out = tf.tile(encoder_out, [beam_size, 1, 1])  # encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = tf.zeros([beam_size, 1, 1, encoder_out.shape[1]], dtype=tf.dtypes.bool)

        decoder_out = self._forward_decoder(encoder_out, encoder_mask, hyps_pad, output_mask)

        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0

        lm_weight = hparams.lm_weight if lm_model is not None else 0.0
        at_weight = hparams.at_weight
        if lm_weight > 0.0:
            lm_logits = lm_model.forward(hyps_pad, hyps_pad_len, training=False)
            lm_logprob = tf.nn.log_softmax(lm_logits)

        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w] * at_weight
                if lm_weight > 0.0:
                    score += lm_logprob[i][j][w] * lm_weight
            score += decoder_out[i][len(hyp[0])][self.eos] * at_weight
            if lm_weight > 0.0:
                score += lm_logprob[i][len(hyp[0])][self.eos] * lm_weight

            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        return tf.convert_to_tensor(hyps[best_index][0])[tf.newaxis, :]  # hyps[best_index][0],

    def freeze_beam_search(self, samples, beam_size=1, hparams=None, lm_model=None):
        """ beam search for freeze only support batch=1

        Args:
            samples: the data source to be decoded
            beam_size: beam size
        """
        x0 = samples["input"]
        batch_size = tf.shape(x0)[0]
        x = self.x_net(x0, training=False)
        input_length = self.x_net.compute_logit_length(samples["input_length"])
        input_mask, _ = create_multihead_mask_u2(x, input_length, None)
        # 1. Encoder
        encoder_output = self.transformer.encoder(x, input_mask, training=False)  # (B, maxlen, encoder_dim)
        maxlen = tf.shape(encoder_output)[1]
        encoder_dim = tf.shape(encoder_output)[2]
        running_size = beam_size

        # repeat for beam_size
        input_mask = tf.tile(input_mask, [1, beam_size, 1, 1])
        input_mask = tf.transpose(input_mask, [1, 0, 2, 3])

        encoder_output = tf.tile(encoder_output, [1, beam_size, 1])
        encoder_output = tf.transpose(encoder_output, [1, 0, 2])

        hyps = tf.ones([running_size, 1], dtype=tf.int32) * self.sos  # (B*N, max_len)
        scores = tf.constant([[0.0] + [-float('inf')] * (beam_size - 1)], shape=(beam_size, 1), dtype=tf.float32)
        scores = tf.tile(scores, [batch_size, 1])

        end_flag = tf.zeros_like(scores, dtype=tf.int32)
        # 2. Decoder forward step by step
        for step in tf.range(1, maxlen + 1):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[
                    (hyps, tf.TensorShape([None, None])),
                    (scores, tf.TensorShape([None, None])),
                    (end_flag, tf.TensorShape([None, None]))
                ])
            if tf.reduce_sum(end_flag) == running_size:
                break
            # 2.1 Forward decoder step
            output_mask = generate_square_subsequent_mask_u2(step)

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
            scores = tf.reshape(scores, [batch_size, beam_size * beam_size])

            scores, best_k_index = tf.math.top_k(scores, k=beam_size)
            scores = tf.reshape(scores, [batch_size * beam_size, 1])
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
            best_k_pred = tf.reshape(best_k_pred, [batch_size * beam_size, 1])

            last_best_k_index = batch_index * beam_size + row_index
            last_best_k_index = tf.reshape(last_best_k_index, [running_size, 1])
            last_best_k_hyps = tf.gather(hyps, last_best_k_index, axis=0)
            last_best_k_hyps = tf.reshape(last_best_k_hyps, [running_size, -1])
            hyps = tf.concat([last_best_k_hyps, best_k_pred], axis=1)
            # 2.7 Update end flag
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

    def beam_search(self, samples, hparams, lm_model=None):
        """ batch beam search for transformer model

        Args:
            samples: the data source to be decoded
            beam_size: beam size
            lm_model: rnnlm that used for beam search

        """
        x0 = samples["input"]
        batch_size = tf.shape(x0)[0]

        tmp_masks = tf.zeros((1, 1, 1, tf.shape(x0)[1]), dtype=tf.bool)
        x, pos_emb, x_mask = self.x_net(x0, tmp_masks)
        input_length = self.x_net.compute_logit_length(samples["input_length"])
        input_mask, _ = create_multihead_mask_u2(x, input_length, None)
        # # 1. Encoder
        # encoder_output = self.transformer.encoder(x, input_mask, training=False)  # (B, maxlen, encoder_dim)
        speech = samples["input"]
        speech_lengths = samples["input_length"]
        beam_size, ctc_weight, simulate_streaming, decoding_chunk_size, num_decoding_left_chunks = \
            hparams.beam_size, hparams.ctc_weight, hparams.simulate_streaming, hparams.decoding_chunk_size, hparams.num_decoding_left_chunks

        if batch_size > 1:
            simulate_streaming = False
        encoder_output, encoder_mask = self._forward_encoder(speech, speech_lengths, training=False,
                                                          simulate_streaming=simulate_streaming,
                                                          decoding_chunk_size=decoding_chunk_size,
                                                          num_decoding_left_chunks=num_decoding_left_chunks)

        maxlen = tf.shape(encoder_output)[1]
        running_size = batch_size * beam_size

        # repeat for beam_size
        batch_shape, b, c, encoder_len = input_mask.get_shape().as_list()
        input_mask = tf.tile(input_mask, [1, beam_size, 1, 1])
        input_mask = tf.reshape(input_mask, [batch_shape * beam_size, b, c, encoder_len])

        batch_shape, encoder_len, encoder_dim = encoder_output.get_shape().as_list()
        encoder_output = tf.tile(encoder_output, [1, beam_size, 1])
        encoder_output = tf.reshape(encoder_output, [batch_shape * beam_size, encoder_len, encoder_dim])

        hyps = tf.ones([running_size, 1], dtype=tf.int32) * self.sos  # (B*N, max_len)
        scores = tf.constant([[0.0] + [-float('inf')] * (beam_size - 1)], shape=(beam_size, 1), dtype=tf.float32)
        scores = tf.tile(scores, [batch_size, 1])

        end_flag = tf.zeros_like(scores, dtype=tf.int32)
        # 2. Decoder forward step by step
        ctc_states = None
        for step in tf.range(1, maxlen + 1):
            if tf.reduce_sum(end_flag) == running_size:
                break
            # 2.1 Forward decoder step
            output_mask = generate_square_subsequent_mask_u2(step)

            y0 = self.y_net(hyps, training=False)

            decoder_outputs = self.transformer.decoder(
                y0,
                encoder_output,
                tgt_mask=output_mask,
                memory_mask=input_mask,  # (beam, 1, 1, L_encode)
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
                lm_logits = lm_model.call(hyps, hyps_len)
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


class SpeechTransformerU2(SpeechU2):
    """ U2 implementation of a SpeechTransformer. Model mainly consists of three parts:
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
        "max_position": 5000,
        "dropout_rate": 0.1,
        "use_dynamic_chunk": True,   # whether use dynamic encoder chunk for training
        "use_dynamic_left_chunk": False,    # you can choose dynamic encoder left chunk for training when use dynamic chunk
        "static_chunk_size": -1,    # if static_chunk_size > 0, encoder chunk size will be static_chunk_size, when dynamic chunk is turn off
        "schedual_sampling_rate": 0.9,
        "label_smoothing_rate": 0.0,
        "unidirectional": False,
        "look_ahead": 0,
        "conv_module_kernel_size": 0
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__(data_descriptions, config)
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        layers = tf.keras.layers
        # for the x_net
        pos_encoding = PositionalEncodingU2(self.hparams.d_model, self.hparams.max_position, self.hparams.dropout_rate)
        self.x_net = Conv2dSubsampling4(self.hparams.num_filters, self.hparams.d_model,
                                        self.feats_dim, 0, pos_encoding)

        # y_net for target
        input_labels = layers.Input(shape=data_descriptions.sample_shape["output"], dtype=tf.int32)
        inner = layers.Embedding(self.num_class, self.hparams.d_model)(input_labels)
        inner = PositionalEncoding(self.hparams.d_model, max_position=self.hparams.max_position, scale=True)(inner)
        inner = layers.Dropout(self.hparams.dropout_rate)(inner)
        self.y_net = tf.keras.Model(inputs=input_labels, outputs=inner, name="y_net")  # TODO:need change
        print(self.y_net.summary())

        # transformer layer
        self.transformer = TransformerU2(
            self.hparams.d_model,
            self.hparams.num_heads,
            self.hparams.num_encoder_layers,
            self.hparams.num_decoder_layers,
            self.hparams.dff,
            self.hparams.dropout_rate,
            unidirectional=self.hparams.unidirectional,
            look_ahead=self.hparams.look_ahead,
            conv_module_kernel_size=self.hparams.conv_module_kernel_size,
            use_dynamic_chunk=self.use_dynamic_chunk
        )
        # last layer for output
        self.final_layer = layers.Dense(self.num_class, input_shape=(self.hparams.d_model,))


class SpeechConformerU2(SpeechU2):
    """
    Conformer-U2
    """
    default_config = {
        "return_encoder_output": False,
        "num_filters": 512,
        "d_model": 512,
        "num_heads": 8,
        "cnn_module_kernel": 15,
        "num_encoder_layers": 12,
        "num_decoder_layers": 6,
        "dff": 1280,
        "max_position": 800,
        "dropout_rate": 0.1,
        "use_dynamic_chunk": True,  # whether use dynamic encoder chunk for training
        "use_dynamic_left_chunk": False,    # you can choose dynamic encoder left chunk for training when use dynamic chunk
        "static_chunk_size": -1,    # if static_chunk_size > 0, encoder chunk size will be static_chunk_size, when dynamic chunk is turn off
        "schedual_sampling_rate": 0.9,
        "label_smoothing_rate": 0.0,
        "unidirectional": False,
        "look_ahead": 0,
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__(data_descriptions, config)

        # for the x_net
        layers = tf.keras.layers
        pos_encoding = PositionalEncodingU2(self.hparams.d_model, self.hparams.max_position, self.hparams.dropout_rate)
        self.x_net = Conv2dSubsampling4(self.hparams.num_filters, self.hparams.d_model,
                                        self.feats_dim, 0, pos_encoding)

        # y_net for target
        input_labels = layers.Input(shape=data_descriptions.sample_shape["output"], dtype=tf.int32)
        inner = layers.Embedding(self.num_class, self.hparams.d_model)(input_labels)
        inner = PositionalEncoding(self.hparams.d_model, max_position=self.hparams.max_position, scale=True)(inner)
        inner = layers.Dropout(self.hparams.dropout_rate)(inner)
        self.y_net = tf.keras.Model(inputs=input_labels, outputs=inner, name="y_net")
        print(self.y_net.summary())

        # transformer layer
        self.transformer = ConformerU2(
            self.hparams.d_model,
            self.hparams.num_heads,
            self.hparams.cnn_module_kernel,
            self.hparams.num_encoder_layers,
            self.hparams.num_decoder_layers,
            self.hparams.dff,
            self.hparams.dropout_rate,
            unidirectional=self.hparams.unidirectional,
            look_ahead=self.hparams.look_ahead,
            use_dynamic_chunk=self.use_dynamic_chunk
        )

        # last layer for output
        self.final_layer = layers.Dense(self.num_class, input_shape=(self.hparams.d_model,))

        # some temp function
        self.random_num = tf.random_uniform_initializer(0, 1)