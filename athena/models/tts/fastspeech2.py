# Copyright (C) 2022 ATHENA AUTHORS; Cheng Wen
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
import tensorflow as tf

from athena.models.base import BaseModel
from athena.utils.hparam import register_and_parse_hparams
from athena.layers.transformer import TransformerEncoderLayer, TransformerEncoder
from athena.layers.commons import ScaledPositionalEncoding
from athena.utils.misc import create_multihead_mask
from athena.loss import FastSpeech2Loss
from .fastspeech import FastSpeech, LengthRegulator


class FastSpeech2(FastSpeech):
    """
    Reference: FastSpeech 2: Fast and High-Quality End-to-End Text to Speech
    """
    default_config = {
        "d_model": 384,
        "num_heads": 4,
        "dff": 1536,
        "rate": 0.1,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "ffn_kernels": 3,
        "batch_norm_position": "before",
        "conv_module_kernel_size": 0,

        "variant_predictor_dropout_rate": 0.5,
        "variant_predictor_layers": 2,
        "variant_predictor_channels": 256,
        "variant_predictor_kernel": 3,
        "duration_predictor_offset": 1.0,

        "postnet_layers": 5,
        "postnet_kernel": 5,
        "postnet_filters": 256,
        "speaker_embedding_dim": 512,
        "speaker_embedding_integration_type": "add",
        "reduction_factor": 1,

        "alpha": 1.0,
        "clip_max": 100,

        "variant_predictor_loss_weight": 1.0,
        "teacher_guide_loss": False
        }

    def __init__(self, data_descriptions, config=None):
        self.default_config.update(config)
        current_default_config = self.default_config
        self.default_config = super().default_config
        self.default_config.update(current_default_config)
        super().__init__(data_descriptions)
        self.default_config = current_default_config
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.f0_predictor = VariantPredictor(
            self.hparams, name="f0_predictor"
        )
        self.energy_predictor = VariantPredictor(
            self.hparams, name="energy_predictor"
        )
        self.duration_predictor = VariantPredictor(
            self.hparams, name="duration_predictor"
        )
         # define f0_embeddings and energy_embeddings
        self.f0_embeddings = tf.keras.layers.Conv1D(
            filters= self.hparams.d_model,
            kernel_size=9,
            padding="same",
            name="f0_embeddings",
        )
        self.f0_dropout = tf.keras.layers.Dropout(0.5)
        self.energy_embeddings = tf.keras.layers.Conv1D(
            filters=self.hparams.d_model,
            kernel_size=9,
            padding="same",
            name="energy_embeddings",
        )
        self.energy_dropout = tf.keras.layers.Dropout(0.5)
        self.duration_calculator = DurationCalculator()
        # define loss function and metrics
        self.loss_function = FastSpeech2Loss(self.hparams.variant_predictor_loss_weight)
        self.metric = tf.keras.metrics.Mean(name="AverageLoss")

    def call(self, samples, training: bool = None):
        x0 = self.x_net(samples['input'], training=training)
        _, input_mask = create_multihead_mask(None, None, samples['input'], reverse=True)
        encoder_output = self.encoder(x0, input_mask, training=training) # [batch, x_steps, d_model]
        batch = tf.shape(encoder_output)[0]
        if self.n_speakers > 1:
            speaker = tf.reshape(samples['speaker'], [batch])
            speaker_embedding = self.speaker_embedding(speaker)
            speaker_embedding = tf.nn.l2_normalize(speaker_embedding, axis=-1)
            if self.hparams.speaker_embedding_integration_type == "add":
                speaker_embedding = tf.reshape(speaker_embedding,[batch, 1, self.hparams.speaker_embedding_dim])
                speaker_embedding = self.speaker_fc(speaker_embedding)
                encoder_output += speaker_embedding
            else:
                x_steps = tf.shape(encoder_output)[1]
                speaker_embedding = tf.expand_dims(speaker_embedding, 1)
                speaker_embedding_expand = tf.tile(speaker_embedding, (1, x_steps, 1)) #shape:[batch, x_steps, speaker_embdedding_dim]
                encoder_output = tf.concat([encoder_output, speaker_embedding_expand], axis = 2)
                encoder_output = self.speaker_fc(encoder_output)
        #dur
        teacher_outs, duration_indexes, duration_sequences = self.duration_calculator(samples)
        pred_duration_sequences = self.duration_predictor(encoder_output, training=training)
        #f0 embedding
        f0_gts = tf.expand_dims(samples["f0"], 2) # [batch_size, mel_length, 1]
        f0_embedding = self.f0_embeddings(f0_gts)
        f0_embedding = self.f0_dropout(f0_embedding, training=training)
        #energy embedding
        energy_gts = tf.expand_dims(samples["energy"], 2)
        energy_embedding = self.energy_embeddings(energy_gts)
        energy_embedding = self.energy_dropout(energy_embedding)

        pred_f0 = self.f0_predictor(encoder_output, training=training)
        pred_energy = self.energy_predictor(encoder_output, training=training)
        output_length = samples['output_length']

        encoder_output += f0_embedding + energy_embedding
        before_outs, after_outs = self._feedforward_decoder(encoder_output, duration_indexes,
                                                            duration_sequences, output_length,
                                                            training=training)
        return before_outs, teacher_outs, after_outs, duration_sequences, pred_duration_sequences, pred_f0, pred_energy


    def synthesize(self, samples):
        x0 = self.x_net(samples['input'], training=False)
        _, input_mask = create_multihead_mask(None, None, samples['input'], reverse=True)
        encoder_output = self.encoder(x0, input_mask, training=False) # [batch, x_steps, d_model]
        batch = tf.shape(encoder_output)[0]
        if self.n_speakers > 1:
            speaker = tf.reshape(samples['speaker'], [batch])
            speaker_embedding = self.speaker_embedding(speaker)
            speaker_embedding = tf.nn.l2_normalize(speaker_embedding, axis=-1)
            if self.hparams.speaker_embedding_integration_type == "add":
                speaker_embedding = tf.reshape(speaker_embedding,[batch, 1, self.hparams.speaker_embedding_dim])
                speaker_embedding = self.speaker_fc(speaker_embedding)
                encoder_output += speaker_embedding
            else:
                x_steps = tf.shape(encoder_output)[1]
                speaker_embedding = tf.expand_dims(speaker_embedding, 1)
                speaker_embedding_expand = tf.tile(speaker_embedding, (1, x_steps, 1)) #shape:[batch, x_steps, speaker_embdedding_dim]
                encoder_output = tf.concat([encoder_output, speaker_embedding_expand], axis = 2)
                encoder_output = self.speaker_fc(encoder_output)

        duration_sequences = self.duration_predictor(encoder_output, training=False)
        duration_sequences = tf.cast(tf.clip_by_value(tf.math.round(
            tf.exp(duration_sequences) - self.hparams.duration_predictor_offset),
            0.0, tf.cast(self.hparams.clip_max, dtype=tf.float32)), dtype=tf.int32)

        f0_sequences = self.f0_predictor(encoder_output, training=False)
        energy_sequences = self.energy_predictor(encoder_output, training=False)
        f0_embedding = self.f0_dropout(
            self.f0_embeddings(tf.expand_dims(f0_sequences, 2)), training=False
        )
        energy_embedding = self.energy_dropout(
            self.energy_embeddings(tf.expand_dims(energy_sequences, 2)), training=False
        )
        encoder_output += f0_embedding + energy_embedding
        _, after_outs = self._feedforward_decoder(encoder_output, None, duration_sequences, None,
                                                  training=False)
        return after_outs, None


class DurationCalculator(tf.keras.layers.Layer):
    """Calculate duration and outputs based on teacher model
    """

    def __init__(self, teacher_model=None, teacher_type=None):
        """Initialize duration calculator
        """
        super().__init__()

    def call(self, samples):
        """
        Args:
            samples: samples from dataset
        Returns::

            durations: Batch of durations shape: [batch, max_input_length).
        """
        y_steps = tf.reduce_max(samples['output_length'])
        x_steps = tf.reduce_max(samples['input_length'])
        teacher_outs = None
        duration_index = samples['duration']
        weights_argmax = duration_index[:, :y_steps] # [batch, y_steps]
        if tf.shape(weights_argmax)[1] < y_steps:
            padding = tf.zeros([tf.shape(weights_argmax)[0], y_steps - tf.shape(weights_argmax)[1]],
                                dtype=tf.int32)
            weights_argmax = tf.concat([weights_argmax, padding], axis=1)
        output_mask = tf.sequence_mask(samples['output_length'], y_steps)
        output_mask = tf.cast(tf.logical_not(output_mask), dtype=tf.int32) * -10000
        weights_argmax += output_mask

        def count_duration(index):
            duration = tf.cast(tf.equal(weights_argmax, index), dtype=tf.int32)
            duration = tf.reduce_sum(duration, axis=1) # [batch]
            duration = tf.cast(duration, dtype=tf.int32)
            return duration

        indexes = tf.range(x_steps)
        durations = tf.map_fn(count_duration, indexes, parallel_iterations=128)
        durations = tf.transpose(durations, [1, 0]) # [batch, max_input_length]
        return teacher_outs, weights_argmax, durations

class  VariantPredictor(tf.keras.layers.Layer):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        # define variant predictor
        layers = tf.keras.layers
        input_features = layers.Input(shape=tf.TensorShape([None, hparams.d_model]),
                                       dtype=tf.float32)
        inner = input_features
        for _ in range(hparams.variant_predictor_layers):
            inner = layers.Conv1D(
                filters=hparams.variant_predictor_channels,
                kernel_size=hparams.variant_predictor_kernel,
                strides=1,
                padding="same",
                use_bias = False,
                data_format = "channels_last")(inner)
            inner = layers.ReLU()(inner)
            inner = layers.LayerNormalization()(inner)
            inner = layers.Dropout(hparams.variant_predictor_dropout_rate)(inner)
        inner = layers.Dense(1)(inner) # [batch, expanded_length, 1]
        inner = tf.squeeze(inner, axis=-1)
        self.variant_predictor = tf.keras.Model(inputs=input_features, outputs=inner,
                                               name="variant_predictor")
        print(self.variant_predictor.summary())
        self.hparams = hparams

    def call(self, inputs, training=False):
        encoder_hidden_states = inputs
        outputs = self.variant_predictor(encoder_hidden_states)
        return outputs

