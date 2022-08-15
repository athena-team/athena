# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Wubo Li; Chaoyang Mei; Jianwei Sun
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
# Only support eager mode and TF>=2.0.0
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes
""" a implementation of multi-task model with attention and ctc loss """
from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense
from athena.models.base import BaseModel
from athena.loss import CTCLoss
from athena.metrics import CTCAccuracy
from .speech_transformer import SpeechTransformer
from .speech_conformer import SpeechConformer
from .speech_u2 import SpeechTransformerU2, SpeechConformerU2
from athena.utils.hparam import register_and_parse_hparams


class MtlTransformerCtc(BaseModel):
    """ In speech recognition, adding CTC loss to Attention-based seq-to-seq model is known to
    help convergence. It usually gives better results than using attention alone.
    """

    SUPPORTED_MODEL = {
        "speech_transformer": SpeechTransformer,
        "speech_conformer": SpeechConformer,
        "speech_transformer_u2": SpeechTransformerU2,
        "speech_conformer_u2": SpeechConformerU2,
    }

    default_config = {
        "model": "speech_transformer",
        "model_config": {"return_encoder_output": True},
        "mtl_weight": 0.5
    }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.num_class = data_descriptions.num_class + 1
        self.sos = self.num_class - 1
        self.eos = self.num_class - 1

        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        
        assert 0.0 < self.hparams.mtl_weight <= 1.0
        self.mtl_weight = self.hparams.mtl_weight
        self.ctc_weight = 1.0 - self.mtl_weight
        self.loss_function = CTCLoss(blank_index=-1)
        self.ctc_metric = CTCAccuracy()
        self.model = self.SUPPORTED_MODEL[self.hparams.model](
            data_descriptions, self.hparams.model_config
        )
        self.metric = self.ctc_metric  # self.model.metric
        self.ctc_final_layer = Dense(self.num_class)

    def call(self, samples, training=None):
        """ call function in keras layers """
        attention_logits, encoder_output = self.model(samples, training=training)
        ctc_logits = None
        if self.ctc_weight > 0.0:
            ctc_logits = self.ctc_final_layer(encoder_output, training=training)            
        return attention_logits, ctc_logits

    def get_loss(self, outputs, samples, training=None):
        """ get loss used for training """
        attention_logits, ctc_logits = outputs
        
        main_loss, metrics = self.model.get_loss(attention_logits, samples, training=training)
        
        extra_loss = 0.0
        if self.ctc_weight > 0.0:
            logit_length = self.compute_logit_length(samples["input_length"])
            extra_loss = self.loss_function(ctc_logits, samples, logit_length)
            self.ctc_metric(ctc_logits, samples, logit_length)
            metrics[self.ctc_metric.name] = self.ctc_metric.result()
        
        loss = self.mtl_weight * main_loss + (1.0 - self.mtl_weight) * extra_loss
        
        return loss, metrics

    def compute_logit_length(self, input_length):
        """ compute the logit length """
        return self.model.compute_logit_length(input_length)

    def reset_metrics(self):
        """ reset the metrics """
        self.ctc_metric.reset_states()
        self.model.reset_metrics()

    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        """ A more general-purpose interface for pretrained model restoration
        Args:
            pretrained_model: checkpoint path of mpc model
            model_type: the type of pretrained model to restore
        """
        self.model.restore_from_pretrained_model(pretrained_model, model_type)

    def _forward_encoder_log_ctc(self, samples, training: bool = None):
        predictions, input_mask = self.model._forward_encoder_log_ctc(samples, self.ctc_final_layer, training)

        return predictions, input_mask

    def decode(self, samples, hparams, lm_model=None):
        """
        Initialization of the model for decoding,
        decoder is called here to create predictions

        Args:
            samples: the data source to be decoded
            hparams: decoding configs are included here
            lm_model: lm model
        Returns::

            predictions: the corresponding decoding results
        """
        if hparams.decoder_type == "attention_rescoring":
            predictions = self.model.attention_rescoring(samples, hparams, self.ctc_final_layer,
                                                         lm_model=lm_model)
        elif hparams.decoder_type == "ctc_prefix_beam_search":
            predictions = self.model.ctc_prefix_beam_search(samples, hparams, self.ctc_final_layer)
            # predictions = self.ctc_prefix_beam_search(samples, hparams, 10, -1)
        elif hparams.decoder_type == "attention_decoder":
            predictions = self.model.beam_search(samples, hparams, lm_model)
        else:
            raise NotImplementedError("Decoder type only support: 1.attention_rescoring 2.ctc_prefix_beam_search "
                                      "3.attention_decoder.")
        return predictions

    def enable_tf_funtion(self):
        self.model.enable_tf_funtion()

    @tf.function
    def ctc_forward_chunk_freeze(self, encoder_out):
        ctc_probs = tf.nn.log_softmax(self.ctc_final_layer(encoder_out), axis=2)
        return {'ctc_probs': ctc_probs,}

    @tf.function
    def encoder_ctc_forward_chunk_freeze(self, chunk_xs, offset, required_cache_size,
                                     subsampling_cache, elayers_output_cache, conformer_cnn_cache, ):
        (encoder_out, offset, subsampling_cache, elayers_output_cache, conformer_cnn_cache) = \
            self.model._encoder_forward_chunk(chunk_xs, offset, required_cache_size,
                                              subsampling_cache, elayers_output_cache, conformer_cnn_cache, )
        ctc_probs = tf.nn.log_softmax(self.ctc_final_layer(encoder_out), axis=2)
        return {'ctc_probs': ctc_probs,
                'encoder_out': encoder_out,
                'offset': offset,
                'subsampling_cache': subsampling_cache,
                'elayers_output_cache': elayers_output_cache,
                'conformer_cnn_cache': conformer_cnn_cache,}

    @tf.function
    def encoder_forward_chunk_freeze(self, chunk_xs, offset, required_cache_size,
                                     subsampling_cache, elayers_output_cache, conformer_cnn_cache, ):
        (encoder_out, offse, subsampling_cache, elayers_output_cache, conformer_cnn_cache) = \
            self.model._encoder_forward_chunk(chunk_xs, offset, required_cache_size,
                                              subsampling_cache, elayers_output_cache, conformer_cnn_cache, )
        return {'encoder_out': encoder_out,
                'offse': offse,
                'subsampling_cache': subsampling_cache,
                'elayers_output_cache': elayers_output_cache,
                'conformer_cnn_cache': conformer_cnn_cache,}

    @tf.function
    def get_subsample_rate(self):
        return {'subsample_rate': self.model.x_net.subsampling_rate,}

    @tf.function
    def get_init(self):
        subsampling_cache, elayers_output_cache, conformer_cnn_cache, right_context, subsampling = self.model.get_encoder_init_input()
        return {
            "subsampling_cache": subsampling_cache,
            "elayers_output_cache": elayers_output_cache,
            "conformer_cnn_cache": conformer_cnn_cache,
            "right_context": right_context,
            "subsampling": subsampling,
        }

    def encoder_forward_chunk_by_chunk_freeze(
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
            speech (tf.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.model.transformer.encoder.static_chunk_size > 0 or self.model.transformer.encoder.use_dynamic_chunk
        subsampling = self.model.x_net.subsampling_rate
        context = self.model.x_net.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = speech.shape[1]
        subsampling_cache = tf.zeros([1, 0, self.model.hparams.d_model], dtype=tf.float32)
        elayers_output_cache = tf.zeros([self.model.hparams.num_encoder_layers, 1, 0, self.model.hparams.d_model],
                                        dtype=tf.float32)
        conformer_cnn_cache = tf.zeros([self.model.hparams.num_encoder_layers, 1, 0, self.model.hparams.d_model],
                                       dtype=tf.float32)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = speech[:, cur:end, :]
            (y, subsampling_cache, elayers_output_cache,
             conformer_cnn_cache) = self.encoder_forward_chunk_freeze(chunk_xs, offset,
                                                                      required_cache_size,
                                                                      subsampling_cache,
                                                                      elayers_output_cache,
                                                                      conformer_cnn_cache, )
            outputs.append(y)
            offset += y.shape[1]
        ys = tf.concat(outputs, 1)
        masks = tf.zeros((1, ys.shape[1]), dtype=tf.bool)
        masks = tf.expand_dims(masks, axis=1)
        return ys, masks

    def ctc_prefix_beam_search(
            self, samples, hparams, decoding_chunk_size, num_decoding_left_chunks
    ) -> List[int]:
        speech = samples["input"]
        speech_lengths = samples["input_length"]
        assert speech.shape[0] == speech_lengths.shape[0]
        beam_size = hparams.beam_size
        ctc_probs, encoder_mask = self.encoder_forward_chunk_by_chunk_freeze(speech,
                                                                             decoding_chunk_size=decoding_chunk_size,
                                                                             num_decoding_left_chunks=num_decoding_left_chunks)

        # ctc_probs = tf.nn.log_softmax(ctc_final_layer(encoder_out), axis=2)  # (1, maxlen, vocab_size)
        ctc_probs = tf.transpose(ctc_probs, (1, 0, 2))
        sequence_length = self.model.compute_logit_length(speech_lengths)
        decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(  # (max_time, batch_size, num_classes)
            ctc_probs, sequence_length, beam_width=beam_size, top_paths=beam_size
        )

        # return tf.convert_to_tensor(hyps[0][0])[tf.newaxis, :]
        return decoded[0].values[tf.newaxis, :]
