import tensorflow as tf

from .base import BaseModel
from ..utils.hparam import register_and_parse_hparams
from ..layers.transformer import TransformerEncoderLayer, TransformerEncoder
from ..layers.commons import ScaledPositionalEncoding
from ..utils.misc import create_multihead_mask
from ..loss import FastSpeechLoss
from absl import logging


class FastSpeech(BaseModel):
    """
    Reference: Fastspeech: Fast, robust and controllable text to speech
      (http://papers.nips.cc/paper/8580-fastspeech-fast-robust-and-controllable-text-to-speech.pdf)
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
        "dropout_rate": 0.5,

        "duration_predictor_layers": 2,
        "duration_predictor_channels": 256,
        "duration_predictor_kernel": 3,
        "duration_predictor_offset": 1.0,

        "postnet_layers": 5,
        "postnet_kernel": 5,
        "postnet_filters": 256,

        "reduction_factor": 1,

        "alpha": 1.0,
        "clip_max": 100,

        "duration_predictor_loss_weight": 1.0,
        "teacher_guide_loss": False
        }

    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        self.num_class = data_descriptions.num_class
        self.eos = self.num_class - 1
        self.feat_dim = data_descriptions.feat_dim
        self.reduction_factor = self.hparams.reduction_factor
        self.data_descriptions = data_descriptions

        # for the x_net
        layers = tf.keras.layers
        input_features = layers.Input(shape=data_descriptions.sample_shape["input"], dtype=tf.int32)
        inner = layers.Embedding(self.num_class, self.hparams.d_model)(input_features)
        inner = ScaledPositionalEncoding(self.hparams.d_model)(inner)
        inner = layers.Dropout(self.hparams.rate)(inner)  # self.hparams.rate
        self.x_net = tf.keras.Model(inputs=input_features, outputs=inner, name="x_net")
        print(self.x_net.summary())

        ffn_list = []
        for _ in range(2):
            ffn_list.append(tf.keras.Sequential(
                [
                    layers.Conv1D(
                        filters=self.hparams.dff,
                        kernel_size=self.hparams.ffn_kernels,
                        strides=1,
                        padding="same",
                        use_bias=False,
                        data_format="channels_last"),
                    layers.ReLU(),
                    layers.Dropout(self.hparams.rate),
                    layers.Conv1D(
                        filters=self.hparams.d_model,
                        kernel_size=self.hparams.ffn_kernels,
                        strides=1,
                        padding="same",
                        use_bias=False,
                        data_format="channels_last")
                ]
            ))
        # define encoder form transform.py
        encoder_layers = [
            TransformerEncoderLayer(self.hparams.d_model, self.hparams.num_heads,
                                    self.hparams.dff, self.hparams.rate, ffn=ffn_list[0])
            for _ in range(self.hparams.num_encoder_layers)
        ]
        self.encoder = TransformerEncoder(encoder_layers)

        # define duration predictor
        input_features = layers.Input(shape=tf.TensorShape([None, self.hparams.d_model]),
                                       dtype=tf.float32)
        inner = input_features
        for _ in range(self.hparams.duration_predictor_layers):
            inner = layers.Conv1D(
                filters=self.hparams.duration_predictor_channels,
                kernel_size=self.hparams.duration_predictor_kernel,
                strides=1,
                padding="same",
                use_bias = False,
                data_format = "channels_last")(inner)
            inner = layers.ReLU()(inner)
            inner = layers.LayerNormalization()(inner)
            inner = layers.Dropout(self.hparams.rate)(inner)
        inner = layers.Dense(1)(inner) # [batch, expanded_length, 1]
        inner = tf.squeeze(inner, axis=-1)
        self.duration_predictor = tf.keras.Model(inputs=input_features, outputs=inner,
                                                 name="duration_predictor")
        print(self.duration_predictor.summary())

        # define length regulator
        self.length_regulator = LengthRegulator()

        # for the y_net
        input_features = layers.Input(shape=tf.TensorShape([None, self.hparams.d_model]),
                                      dtype=tf.float32)
        inner = ScaledPositionalEncoding(self.hparams.d_model, max_position=3200)(input_features)
        inner = layers.Dropout(self.hparams.rate)(inner)  # self.hparams.rate
        self.y_net = tf.keras.Model(inputs=input_features, outputs=inner, name="y_net")
        print(self.y_net.summary())

        # define decoder
        decoder_layers = [
            TransformerEncoderLayer(self.hparams.d_model, self.hparams.num_heads,
                                    self.hparams.dff, self.hparams.rate, ffn=ffn_list[1])
            for _ in range(self.hparams.num_decoder_layers)
        ]
        self.decoder = TransformerEncoder(decoder_layers)

        # define feat_out
        self.feat_out = layers.Dense(self.feat_dim * self.reduction_factor, use_bias=False,
                                     name='feat_projection')
        # define postnet
        input_features_postnet = layers.Input(shape=tf.TensorShape([None, self.feat_dim]),
                                              dtype=tf.float32)
        inner = input_features_postnet
        for _ in tf.range(self.hparams.postnet_layers):
            # filters = self.feat_dim if layer == self.hparams.postnet_layers - 1 \
            #    else self.hparams.postnet_filters
            filters = self.hparams.postnet_filters
            inner = layers.Conv1D(
                filters=filters,
                kernel_size=self.hparams.postnet_kernel,
                strides=1,
                padding="same",
                use_bias=False,
                data_format="channels_last",
            )(inner)
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

        self.teacher_model = None
        self.teacher_type = None
        self.duration_calculator = DurationCalculator()

        # define loss function and metrics
        self.loss_function = FastSpeechLoss(self.hparams.duration_predictor_loss_weight,
                                            teacher_guide=self.hparams.teacher_guide_loss)
        self.metric = tf.keras.metrics.Mean(name="AverageLoss")

    def set_teacher_model(self, teacher_model, teacher_type):
        """set teacher model and initialize duration_calculator before training
        
        Args:
            teacher_model: the loaded teacher model
            teacher_type: the model type, e.g., tacotron2, tts_transformer
        """
        self.teacher_model = teacher_model
        self.teacher_type = teacher_type
        self.duration_calculator = DurationCalculator(self.teacher_model, self.teacher_type)

    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        """restore from pretrained model
        
        Args:
            pretrained_model: the loaded pretrained model
            model_type: the model type, e.g: tts_transformer
        """
        if model_type == "":
            return
        if model_type == "tts_transformer":
            logging.info("loading from pretrained tts transformer model")
            self.x_net = pretrained_model.x_net
            self.encoder = pretrained_model.encoder
        else:
            raise ValueError("NOT SUPPORTED")

    def get_loss(self, outputs, samples, training=None):
        """ get loss used for training """
        loss = self.loss_function(outputs, samples)
        total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
        self.metric.update_state(total_loss)
        metrics = {self.metric.name: self.metric.result()}
        return loss, metrics

    def _feedforward_decoder(self, encoder_output, duration_indexes, duration_sequences,
                             output_length, training):
        """feed-forward decoder

        Args:
            encoder_output: encoder outputs, shape: [batch, x_steps, d_model]
            duration_indexes: argmax weights calculated from duration_calculator.
                It is used for training only, shape: [batch, y_steps]
            duration_sequences: It contains duration information for each phoneme,
                shape: [batch, x_steps]
            output_length: the real output length
            training: if it is in the training stage
        Returns::

            before_outs: the outputs before postnet calculation
            after_outs: the outputs after postnet calculation
        """
        # when using real durations, duration_indexes would be None
        if training is True:
            expanded_array = self.length_regulator(encoder_output, duration_indexes,
                                                   output_length=output_length)
            expanded_length = output_length
        else:
            expanded_array = self.length_regulator.inference(encoder_output, duration_sequences,
                                                             alpha=self.hparams.alpha)
            expanded_length = tf.reduce_sum(duration_sequences, axis=1) # [batch]
        expanded_array = tf.ensure_shape(expanded_array, [None, None, self.hparams.d_model])
        expanded_mask, _ = create_multihead_mask(expanded_array, expanded_length, None)
        expanded_output = self.y_net(expanded_array, training=training)
        # decoder_output, shape: [batch, expanded_length, d_model]
        decoder_output = self.decoder(expanded_output, expanded_mask, training=training)
        batch = tf.shape(decoder_output)[0]
        decoder_output = self.feat_out(decoder_output, training=training)
        before_outs = tf.reshape(decoder_output, [batch, -1, self.feat_dim])
        after_outs = before_outs + self.postnet(before_outs, training=training)
        return before_outs, after_outs

    def call(self, samples, training: bool = None):
        x0 = self.x_net(samples['input'], training=training)
        _, input_mask = create_multihead_mask(None, None, samples['input'], reverse=True)
        encoder_output = self.encoder(x0, input_mask, training=training) # [batch, x_steps, d_model]
        teacher_outs, duration_indexes, duration_sequences = self.duration_calculator(samples)
        if teacher_outs is None:
            teacher_outs = tf.zeros_like(samples['output'])
        pred_duration_sequences = self.duration_predictor(encoder_output, training=training)
        output_length = samples['output_length']
        before_outs, after_outs = self._feedforward_decoder(encoder_output, duration_indexes,
                                                            duration_sequences, output_length,
                                                            training=training)
        return before_outs, teacher_outs, after_outs, duration_sequences, pred_duration_sequences

    def synthesize(self, samples):
        x0 = self.x_net(samples['input'], training=False)
        _, input_mask = create_multihead_mask(None, None, samples['input'], reverse=True)
        encoder_output = self.encoder(x0, input_mask, training=False) # [batch, x_steps, d_model]

        duration_sequences = self.duration_predictor(encoder_output, training=False)
        duration_sequences = tf.cast(tf.clip_by_value(tf.math.round(
            tf.exp(duration_sequences) - self.hparams.duration_predictor_offset),
            0.0, tf.cast(self.hparams.clip_max, dtype=tf.float32)), dtype=tf.int32)
        _, after_outs = self._feedforward_decoder(encoder_output, None, duration_sequences, None,
                                                  training=False)
        return after_outs, None


class LengthRegulator(tf.keras.layers.Layer):
    """Length regulator for Fastspeech.
    """

    def __init__(self):
        super().__init__()

    def inference(self, phoneme_sequences, duration_sequences, alpha=1.0):
        """Calculate replicated sequences based on duration sequences

        Args:
            phoneme_sequences: sequences of phoneme features, shape: [batch, x_steps, d_model]
            duration_sequences: durations of each frame, shape: [batch, x_steps]
            alpha: Alpha value to control speed of speech.
        Returns::

            expanded_array: replicated sequences based on durations, shape: [batch, y_steps, d_model]
        """
        if alpha != 1.0:
            duration_sequences = tf.cast(
                tf.math.round(tf.cast(duration_sequences, dtype=tf.float32) * alpha),
                dtype=tf.int32)
        x_steps = tf.shape(phoneme_sequences)[1]
        batch = tf.shape(phoneme_sequences)[0]
        # the duration value of one phoneme at least is one

        # max_duration_i: the max duration of all the indexes
        max_duration_i = tf.reduce_max(duration_sequences)
        background = tf.ones([batch, x_steps, max_duration_i], dtype=tf.int32) * -1

        # durations represents frame size for each index, shape: [batch, x_steps, max_duration_i]
        durations = tf.tile(duration_sequences[:, :, tf.newaxis], [1, 1, max_duration_i])
        duration_order_array = tf.range(max_duration_i)
        duration_order_array = tf.tile(duration_order_array[tf.newaxis, tf.newaxis, :],
                                       [batch, x_steps, 1])
        step_order_array = tf.range(x_steps)
        step_order_array = tf.tile(step_order_array[tf.newaxis, :, tf.newaxis],
                                   [batch, 1, max_duration_i])
        #duration_indexes examples:
        #if we discard the batch dimension, duration_indexes will be like:
        #    [[0, 0, -1, -1],
        #     [1, 1, 1, -1],
        #     [2, -1, -1, -1]]
        #And the corresponding duration sequence is [2, 3, 1]
        duration_indexes = tf.where(duration_order_array < durations, step_order_array, background)
        duration_indexes = tf.reshape(duration_indexes, [batch, -1]) # [batch, x_steps*max_duration_i]

        valid_durations = tf.cast((duration_indexes != -1), dtype=tf.int32)
        total_duration_batches = tf.reduce_sum(valid_durations, axis=-1) # [batch]
        y_steps = tf.reduce_max(total_duration_batches)

        def validate_duration_sequences(batch_i):
            duration_index_per_batch = duration_indexes[batch_i] # [x_step * max_duration_i]
            # it is used to discard the index with the value of -1
            valid_duration_index_per_batch = valid_durations[batch_i] # [x_step * max_duration_i]
            valid_duration_index_per_batch = tf.cast(valid_duration_index_per_batch, dtype=tf.bool)
            duration_index_per_batch = duration_index_per_batch[valid_duration_index_per_batch]
            total_duration_per_batch = total_duration_batches[batch_i]
            padding = tf.ones([y_steps - total_duration_per_batch], dtype=tf.int32) * -1
            valid_duration_seqs = tf.concat([duration_index_per_batch, padding], axis=0) # [y_steps]
            batch_padding = tf.ones([y_steps, 1], dtype=tf.int32) * batch_i # [y_steps]
            valid_duration_seqs = tf.concat([batch_padding, valid_duration_seqs[:, tf.newaxis]],
                                            axis=-1)
            return valid_duration_seqs

        batches = tf.range(batch)
        # duration_indexes, shape: [batch, y_steps, 2]
        batch_duration_indexes = tf.map_fn(validate_duration_sequences, batches, parallel_iterations=128)
        # phoneme_sequences, shape: [batch, x_steps, d_model]
        # expanded_phone_list, shape: [batch, y_steps, d_model]
        expanded_phone_list = tf.gather_nd(phoneme_sequences, batch_duration_indexes)
        return expanded_phone_list

    def call(self, phoneme_sequences, duration_indexes, output_length):
        """Calculate replicated sequences based on duration sequences

        Args:
            phoneme_sequences: sequences of phoneme features, shape: [batch, x_steps, d_model]
            duration_indexes: durations of each frame, shape: [batch, y_steps]
            output_length: shape: [batch]
        Returns::

            expanded_array: replicated sequences based on durations, shape: [batch, y_steps, d_model]
        """
        duration_indexes = tf.expand_dims(duration_indexes, axis=-1) # [batch, y_steps, 1]
        batch = tf.shape(duration_indexes)[0]
        y_steps = tf.shape(duration_indexes)[1]
        batch_dims = tf.range(batch)
        batch_dims = batch_dims[:, tf.newaxis, tf.newaxis]
        batch_dims = tf.tile(batch_dims, [1, y_steps, 1]) # [batch, y_steps, 1]
        duration_indexes = tf.concat([batch_dims, duration_indexes], axis=-1) # [batch, y_steps, 2]
        # expanded_array, shape: [batch, y_steps, d_model]
        expanded_array = tf.gather_nd(phoneme_sequences, duration_indexes)
        output_mask = tf.sequence_mask(output_length, dtype=tf.float32)
        output_mask = tf.expand_dims(output_mask, axis=-1) # [batch, y_steps, 1]
        expanded_array *= output_mask
        return expanded_array


class DurationCalculator(tf.keras.layers.Layer):
    """Calculate duration and outputs based on teacher model
    """

    def __init__(self, teacher_model=None, teacher_type=None):
        """Initialize duration calculator

        Args:
            teacher_model: the pretrained model used to calculate durations and outputs.
            teacher_type: the teacher type
        """
        super().__init__()
        self.teacher_model = teacher_model
        self.teacher_type = teacher_type

    def call(self, samples):
        """
        Args:
            samples: samples from dataset
        Returns::

            durations: Batch of durations shape: [batch, max_input_length).
        """
        y_steps = tf.reduce_max(samples['output_length'])
        x_steps = tf.reduce_max(samples['input_length'])
        batch = tf.shape(samples['input_length'])[0]
        teacher_outs = None
        if self.teacher_type is None:
            weights_argmax = samples['duration']
            if tf.shape(weights_argmax)[1] == 0:
                # for initialization
                weights_argmax = tf.ones([batch, y_steps], dtype=tf.int32)
        elif self.teacher_type == 'tts_transformer':
            teacher_outs, attn_weights = self._calculate_transformer_attentions(samples)
            weights_argmax = tf.cast(tf.argmax(attn_weights, axis=-1), dtype=tf.int32)
        elif self.teacher_type == 'tacotron2':
            teacher_outs, attn_weights = self._calculate_t2_attentions(samples)
            weights_argmax = tf.cast(tf.argmax(attn_weights, axis=-1), dtype=tf.int32)
        else:
            raise ValueError("teacher type not supported")
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

    def _calculate_t2_attentions(self, samples):
        """Calculate outputs and attention weights based on tacotron2
        
        Args:
            samples: samples from dataset
        Returns::
            after_outs: the outputs from the teacher model
            attn_weights: the attention weights from the teacher model
        """
        # attn_weights shape: [batch, y_steps, x_steps]
        _, after_outs, _, attn_weights = self.teacher_model(samples, training=False)
        attn_weights = tf.stop_gradient(attn_weights)
        after_outs = tf.stop_gradient(after_outs)
        return after_outs, attn_weights

    def _calculate_transformer_attentions(self, samples):
        """Calculate outputs and attention weights based on tts transformer

        Args:
            samples: samples from dataset
        Returns::
            after_outs: the outputs from the teacher model
            attn_weights: the attention weights from the teacher model
        """
        _, after_outs, _, attn_weights = self.teacher_model(samples, training=False)
        # attn_weights shape: [layers, batch, heads, y_steps, x_steps]
        attn_weights = tf.convert_to_tensor(attn_weights)

        max_weights = tf.reduce_max(attn_weights, axis=-1)
        max_weights = tf.reduce_mean(max_weights, axis=[1, 3]) # shape: [layers, heads]
        max_weights = tf.reshape(max_weights, [-1])
        best_attention_index = tf.argmax(max_weights)

        batch = tf.shape(attn_weights)[1]
        y_steps = tf.shape(attn_weights)[3]
        x_steps = tf.shape(attn_weights)[4]
        attn_weights = tf.transpose(attn_weights, [0, 2, 1, 3, 4])
        attn_weights = tf.reshape(attn_weights, [-1, batch, y_steps, x_steps])
        max_weights = attn_weights[best_attention_index] # [batch, y_steps, x_steps]
        max_weights = tf.stop_gradient(max_weights)
        after_outs = tf.stop_gradient(after_outs)
        return after_outs, max_weights



