import tensorflow as tf
from .fastspeech import FastSpeech
from ..utils.hparam import register_and_parse_hparams
from ..layers.commons import ScaledPositionalEncoding
from ..utils.misc import create_multihead_mask
from ..loss import FastSpeech2Loss


class FastSpeech2(FastSpeech):
    default_config = {
        "d_model": 384,
        "num_heads": 2,
        "dff": 1536,
        "rate": 0.1,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "ffn_kernels": 9,
        "batch_norm_position": "before",
        "dropout_rate": 0.5,
        "bins": 256,

        "duration_predictor_layers": 2,
        "duration_predictor_channels": 256,
        "duration_predictor_kernel": 3,
        "duration_predictor_offset": 1.0,

        "pitch_predictor_layers": 2,
        "pitch_predictor_channels": 256,
        "pitch_predictor_kernel": 3,
        "pitch_embedding_kernel": 9,
        "pitch_embedding_dropout": 0.5,

        "energy_predictor_layers": 2,
        "energy_predictor_channels": 256,
        "energy_predictor_kernel": 3,
        "energy_embedding_kernel": 9,
        "energy_embedding_dropout": 0.5,

        "postnet_layers": 5,
        "postnet_kernel": 5,
        "postnet_filters": 256,

        "reduction_factor": 1,

        "speed_alpha": 1.0,
        "pitch_alpha": 1.0,
        "energy_alpha": 1.0,
        "clip_max": 100,

        "duration_predictor_loss_weight": 1.0,
        "energy_predictor_loss_weight": 1.0,
        "pitch_predictor_loss_weight": 1.0
        }

    def __init__(self, data_descriptions, config=None):
        self.default_config.update(config)
        current_default_config = self.default_config
        self.default_config = super().default_config
        self.default_config.update(current_default_config)
        super().__init__(data_descriptions)

        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        layers = tf.keras.layers
        # define pitch predictor
        input_features = layers.Input(shape=tf.TensorShape([None, self.hparams.d_model]),
                                       dtype=tf.float32)
        inner = input_features
        for _ in range(self.hparams.pitch_predictor_layers):
            inner = layers.Conv1D(
                filters=self.hparams.pitch_predictor_channels,
                kernel_size=self.hparams.pitch_predictor_kernel,  # 9
                strides=1,
                padding="same",
                use_bias=False,
                data_format="channels_last")(inner)
            inner = layers.ReLU()(inner)
            inner = layers.LayerNormalization()(inner)
            inner = layers.Dropout(self.hparams.dropout_rate)(inner)
        inner = layers.Dense(1)(inner) # [batch, expanded_length, 1]
        inner = tf.squeeze(inner, axis=-1)
        self.pitch_predictor = tf.keras.Model(inputs=input_features, outputs=inner,
                                              name="pitch_predictor")

        # define energy predictor
        input_features = layers.Input(shape=tf.TensorShape([None, self.hparams.d_model]),
                                       dtype=tf.float32)
        inner = input_features
        for _ in range(self.hparams.energy_predictor_layers):
            inner = layers.Conv1D(
                filters=self.hparams.energy_predictor_channels,
                kernel_size=self.hparams.energy_predictor_kernel,  # 9
                strides=1,
                padding="same",
                use_bias=False,
                data_format="channels_last")(inner)
            inner = layers.ReLU()(inner)
            inner = layers.LayerNormalization()(inner)
            inner = layers.Dropout(self.hparams.dropout_rate)(inner)
        inner = layers.Dense(1)(inner) # [batch, expanded_length, 1]
        inner = tf.squeeze(inner, axis=-1)
        self.energy_predictor = tf.keras.Model(inputs=input_features, outputs=inner,
                                               name="energy_predictor")

        self.energy_embeddings = tf.keras.Sequential(
                [
                    layers.Conv1D(
                        filters=self.hparams.d_model,
                        kernel_size=self.hparams.energy_embedding_kernel,
                        strides=1,
                        padding="same",
                        use_bias=False,
                        data_format="channels_last"),
                    layers.Dropout(self.hparams.energy_embedding_dropout),
                ]
            )
        self.pitch_embeddings = tf.keras.Sequential(
                [
                    layers.Conv1D(
                        filters=self.hparams.d_model,
                        kernel_size=self.hparams.pitch_embedding_kernel,
                        strides=1,
                        padding="same",
                        use_bias=False,
                        data_format="channels_last"),
                    layers.Dropout(self.hparams.pitch_embedding_dropout),
                ]
            )

        # define loss function and metrics
        self.loss_function = FastSpeech2Loss(self.hparams.duration_predictor_loss_weight,
                                            teacher_guide=self.hparams.teacher_guide_loss,
                                            pitch_predictor_loss_weight=self.hparams.pitch_predictor_loss_weight,
                                            energy_predictor_loss_weight=self.hparams.energy_predictor_loss_weight)
        self.metric = tf.keras.metrics.Mean(name="AverageLoss")

    def call(self, samples, training: bool = None):
        x0 = self.x_net(samples['input'], training=training)
        _, input_mask = create_multihead_mask(None, None, samples['input'], reverse=True)

        encoder_output = self.encoder(x0, input_mask, training=training) 
        last_encoder_hidden_states = encoder_output[0]
        _, duration_indexes, duration_sequences = self.duration_calculator(samples)

        pred_duration_sequences = self.duration_predictor(encoder_output, training=training)
        
        pred_pitch_sequences = self.pitch_predictor(encoder_output, training=training)
        pred_energy_sequences = self.energy_predictor(encoder_output, training=training)
        
        pitch_embeddings = self.pitch_embeddings(samples['pitch'][:, :, tf.newaxis], training=training)
        energy_embeddings = self.energy_embeddings(samples['energy'][:, :, tf.newaxis], training=training)

        encoder_output += pitch_embeddings + energy_embeddings

        expanded_array, expanded_length = self.length_regulate(encoder_output, duration_indexes, duration_sequences, samples['output_length'], training=training)

        before_outs, after_outs = self._feedforward_decoder(expanded_array, expanded_length,
                                                            training=training)
        return before_outs, None, after_outs, duration_sequences, [pred_duration_sequences, pred_pitch_sequences,pred_energy_sequences]                                      

    def synthesize(self, samples, speed_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
        x0 = self.x_net(samples['input'], training=False)
        _, input_mask = create_multihead_mask(None, None, samples['input'], reverse=True)
        encoder_output = self.encoder(x0, input_mask, training=False)  # [batch, x_steps, d_model]
        duration_sequences = self.duration_predictor(encoder_output, training=False)
        duration_sequences *= speed_alpha
        duration_sequences = tf.cast(tf.clip_by_value(tf.math.round(
            tf.exp(duration_sequences) - self.hparams.duration_predictor_offset),
            0.0, tf.cast(self.hparams.clip_max, dtype=tf.float32)), dtype=tf.int32)
        

        pitch_sequences = self.pitch_predictor(encoder_output, training=False)
        pitch_sequences *= pitch_alpha
        energy_sequences = self.energy_predictor(encoder_output, training=False)
        energy_sequences *= energy_alpha

        pitch_embedding = self.pitch_embeddings(pitch_sequences[:, :, tf.newaxis], training=False)
        energy_embedding = self.energy_embeddings(energy_sequences[:, :, tf.newaxis], training=False)
        encoder_output += pitch_embedding + energy_embedding
        expanded_array, expanded_length = self.length_regulate(encoder_output, None,
                                                               duration_sequences, None,
                                                               training=False)

        before_outs, after_outs = self._feedforward_decoder(expanded_array, expanded_length,
                                                  training=False)
        if self.hparams.postnet_layers > 0:
            return after_outs
        else:
            return before_outs
