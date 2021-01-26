# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Jianwei Sun; Ruixiong Zhang; Dongwei Jiang; Chunxin Xiao
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
# pylint: disable=arguments-differ
# pylint: disable=no-member
""" high-level abstraction of different stages in speech processing """

import warnings
import time
import os
import tensorflow as tf
from absl import logging
try:
    import horovod.tensorflow as hvd
except ImportError:
    print("There is some problem with your horovod installation. But it wouldn't affect single-gpu training")
import pyworld
import numpy as np
import librosa
from .utils.hparam import register_and_parse_hparams
from .utils.metric_check import MetricChecker
from .utils.misc import validate_seqs
from .metrics import CharactorAccuracy
from .tools.vocoder import GriffinLim
from .tools.beam_search import BeamSearchDecoder

try:
    from pydecoders import WFSTDecoder
except ImportError:
    print("pydecoder is not installed, this will only affect WFST decoding")


class BaseSolver(tf.keras.Model):
    """Base Solver.
    """
    default_config = {
        "clip_norm": 100.0,
        "log_interval": 10,
        "enable_tf_function": True
    }
    def __init__(self, model, optimizer, sample_signature, eval_sample_signature=None,
                 config=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.optimizer = optimizer
        self.metric_checker = MetricChecker(self.optimizer)
        self.sample_signature = sample_signature
        self.eval_sample_signature = eval_sample_signature
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

    @staticmethod
    def initialize_devices(solver_gpus=None):
        """ initialize hvd devices, should be called firstly """
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # means we're running in GPU mode
        if len(gpus) != 0:
            # If the list of solver gpus is empty, the first gpu will be used.
            if len(solver_gpus) == 0:
                solver_gpus.append(0)
            assert len(gpus) >= len(solver_gpus)
            for idx in solver_gpus:
                tf.config.experimental.set_visible_devices(gpus[idx], "GPU")

    @staticmethod
    def clip_by_norm(grads, norm):
        """ clip norm using tf.clip_by_norm """
        if norm <= 0:
            return grads
        grads = [
            None if gradient is None else tf.clip_by_norm(gradient, norm)
            for gradient in grads
        ]
        return grads

    def train_step(self, samples):
        """ train the model 1 step """
        with tf.GradientTape() as tape:
            # outputs of a forward run of model, potentially contains more than one item
            outputs = self.model(samples, training=True)
            loss, metrics = self.model.get_loss(outputs, samples, training=True)
            total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads = self.clip_by_norm(grads, self.hparams.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, metrics

    def train(self, dataset, total_batches=-1):
        """ Update the model in 1 epoch """
        train_step = self.train_step
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            train_step = tf.function(train_step, input_signature=self.sample_signature)
        for batch, samples in enumerate(dataset.take(total_batches)):
            # train 1 step
            samples = self.model.prepare_samples(samples)
            loss, metrics = train_step(samples)
            if batch % self.hparams.log_interval == 0:
                logging.info(self.metric_checker(loss, metrics))
                self.model.reset_metrics()

    def evaluate_step(self, samples):
        """ evaluate the model 1 step """
        # outputs of a forward run of model, potentially contains more than one item
        outputs = self.model(samples, training=False)
        loss, metrics = self.model.get_loss(outputs, samples, training=False)
        return loss, metrics

    def evaluate(self, dataset, epoch):
        """ evaluate the model """
        loss_metric = tf.keras.metrics.Mean(name="AverageLoss")
        loss, metrics = None, None
        evaluate_step = self.evaluate_step
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            evaluate_step = tf.function(evaluate_step, input_signature=self.eval_sample_signature)
        self.model.reset_metrics()  # init metric.result() with 0
        for batch, samples in enumerate(dataset):
            samples = self.model.prepare_samples(samples)
            loss, metrics = evaluate_step(samples)
            if batch % self.hparams.log_interval == 0:
                logging.info(self.metric_checker(loss, metrics, -2))
            total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
            loss_metric.update_state(total_loss)
        logging.info(self.metric_checker(loss_metric.result(), metrics, evaluate_epoch=epoch))
        self.model.reset_metrics()
        return loss_metric.result(), metrics

class HorovodSolver(BaseSolver):
    """ A multi-processer solver based on Horovod """

    @staticmethod
    def initialize_devices(solver_gpus=None):
        """initialize hvd devices, should be called firstly

        For examples, if you have two machines and each of them contains 4 gpus:
        1. run with command horovodrun -np 6 -H ip1:2,ip2:4 and set solver_gpus to be [0,3,0,1,2,3],
           then the first gpu and the last gpu on machine1 and all gpus on machine2 will be used.
        2. run with command horovodrun -np 6 -H ip1:2,ip2:4 and set solver_gpus to be [],
           then the first 2 gpus on machine1 and all gpus on machine2 will be used.

        Args:
            solver_gpus ([list]): a list to specify gpus being used.

        Raises:
            ValueError: If the list of solver gpus is not empty,
                        its size should not be smaller than that of horovod configuration.
        """
        hvd.init()
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if len(gpus) != 0:
            if len(solver_gpus) > 0:
                if len(solver_gpus) < hvd.size():
                    raise ValueError("If the list of solver gpus is not empty, its size should " +
                                     "not be smaller than that of horovod configuration")
                tf.config.experimental.set_visible_devices(gpus[solver_gpus[hvd.rank()]], "GPU")
            # If the list of solver gpus is empty, the first hvd.size() gpus will be used.
            else:
                tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")

    def train_step(self, samples):
        """ train the model 1 step """
        with tf.GradientTape() as tape:
            # outputs of a forward run of model, potentially contains more than one item
            outputs = self.model(samples, training=True)
            loss, metrics = self.model.get_loss(outputs, samples, training=True)
            total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
        # Horovod: add Horovod Distributed GradientTape.
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads = self.clip_by_norm(grads, self.hparams.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, metrics

    def train(self, dataset, total_batches=-1):
        """ Update the model in 1 epoch """
        train_step = self.train_step
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            train_step = tf.function(train_step, input_signature=self.sample_signature)
        for batch, samples in enumerate(dataset.take(total_batches)):
            # train 1 step
            samples = self.model.prepare_samples(samples)
            loss, metrics = train_step(samples)
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            #
            # Note: broadcast should be done after the first gradient step to ensure optimizer
            # initialization.
            if batch == 0:
                hvd.broadcast_variables(self.model.trainable_variables, root_rank=0)
                hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)
            if batch % self.hparams.log_interval == 0 and hvd.rank() == 0:
                logging.info(self.metric_checker(loss, metrics))
                self.model.reset_metrics()

    def evaluate(self, dataset, epoch=0):
        """ evaluate the model """
        loss_metric = tf.keras.metrics.Mean(name="AverageLoss")
        loss, metrics = None, None
        evaluate_step = self.evaluate_step
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            evaluate_step = tf.function(evaluate_step, input_signature=self.eval_sample_signature)
        self.model.reset_metrics()
        for batch, samples in enumerate(dataset):
            samples = self.model.prepare_samples(samples)
            loss, metrics = evaluate_step(samples)
            if batch % self.hparams.log_interval == 0 and hvd.rank() == 0:
                logging.info(self.metric_checker(loss, metrics, -2))
            total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
            loss_metric.update_state(total_loss)
        if hvd.rank() == 0:
            logging.info(self.metric_checker(loss_metric.result(), metrics, evaluate_epoch=epoch))
            self.model.reset_metrics()
        return loss_metric.result(), metrics


class DecoderSolver(BaseSolver):
    """ DecoderSolver
    """
    default_config = {
        "inference_type": "asr",
        "decoder_type": "wfst_decoder",
        "model_avg_num": 1,
        "beam_size": 4,
        "ctc_weight": 0.0,
        "lm_weight": 0.1,
        "lm_type": "",
        "lm_path": None,
        "acoustic_scale": 10.0,
        "max_active": 80,
        "min_active": 0,
        "wfst_beam": 30.0,
        "max_seq_len": 100,
        "wfst_graph": None
    }

    # pylint: disable=super-init-not-called
    def __init__(self, model, data_descriptions=None, config=None):
        super().__init__(model, None, None)
        self.model = model
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        lm_model = None
        if self.hparams.lm_type == "rnn":
            from athena.main import build_model_from_jsonfile
            _, lm_model, _, lm_checkpointer = build_model_from_jsonfile(self.hparams.lm_path)
            lm_checkpointer.restore_from_best()
        if self.hparams.decoder_type == "beam_search_decoder":
            self.decoder = BeamSearchDecoder.build_decoder(self.hparams,
                                                           self.model.num_class,
                                                           self.model.sos,
                                                           self.model.eos,
                                                           self.model.time_propagate,
                                                           lm_model=lm_model)
        elif self.hparams.decoder_type == "wfst_decoder":
            self.decoder = WFSTDecoder(self.hparams.wfst_graph, acoustic_scale=self.hparams.acoustic_scale,
                                       max_active=self.hparams.max_active, min_active=self.hparams.min_active,
                                       beam=self.hparams.wfst_beam, max_seq_len=self.hparams.max_seq_len,
                                       sos=self.model.sos, eos=self.model.eos)
        else:
            raise ValueError("This decoder type is not supported")

    def inference(self, dataset, rank_size=1):
        """ decode the model """
        if dataset is None:
            return
        metric = CharactorAccuracy(rank_size=rank_size)
        for _, samples in enumerate(dataset):
            begin = time.time()
            samples = self.model.prepare_samples(samples)
            predictions = self.model.decode(samples, self.hparams, self.decoder)
            validated_preds = validate_seqs(predictions, self.model.eos)[0]
            validated_preds = tf.cast(validated_preds, tf.int64)
            num_errs, _ = metric.update_state(validated_preds, samples)
            reports = (
                "predictions: %s\tlabels: %s\terrs: %d\tavg_acc: %.4f\tsec/iter: %.4f"
                % (
                    predictions,
                    samples["output"].numpy(),
                    num_errs,
                    metric.result(),
                    time.time() - begin,
                )
            )
            logging.info(reports)
        logging.info("decoding finished")


class SynthesisSolver(BaseSolver):
    """ SynthesisSolver
    """
    default_config = {
        "inference_type": "tts",
        "model_avg_num": 1,
        "gl_iters": 64,
        "synthesize_from_true_fbank": True,
        "output_directory": None
    }

    def __init__(self, model, data_descriptions=None, config=None):
        super().__init__(model, None, None)
        self.model = model
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.feature_normalizer = data_descriptions.feature_normalizer
        self.speakers_ids_dict = data_descriptions.speakers_ids_dict
        self.vocoder = GriffinLim(data_descriptions)
        self.sample_signature = data_descriptions.sample_signature

    def inference(self, dataset, rank_size=1):
        """ synthesize using vocoder on dataset """
        if dataset is None:
            return
        total_elapsed = 0
        total_seconds = 0
        synthesize_step = tf.function(self.model.synthesize, input_signature=self.sample_signature)
        for i, samples in enumerate(dataset):
            samples = self.model.prepare_samples(samples)
            speaker = samples['speaker']
            speaker = self.speakers_ids_dict[int(speaker[0])]
            start = time.time()
            outputs = synthesize_step(samples)
            end = time.time() - start
            total_elapsed += end
            features, _ = outputs
            if self.hparams.synthesize_from_true_fbank:
                samples_outputs = self.feature_normalizer(samples['output'][0],
                                                          speaker, reverse=True)
                self.vocoder(samples_outputs.numpy(), self.hparams, name='true_%s' % str(i))
            features = self.feature_normalizer(features[0], speaker, reverse=True)
            seconds = self.vocoder(features.numpy(), self.hparams, name=i)
            total_seconds += seconds
        logging.info("model computation elapsed: %s\ttotal seconds: %s\tRTF: %.4f"
                     % (total_elapsed, total_seconds, float(total_elapsed / total_seconds)))


class GanSolver(BaseSolver):
    """ Gan Solver.
    """
    default_config = {
        "clip_norm": 100.0,
        "log_interval": 10,
        "enable_tf_function": True
    }

    def __init__(self, model, sample_signature, config=None):
        super().__init__(model, None, sample_signature)
        self.model = model
        self.metric_checker = MetricChecker(self.model.get_stage_model("generator").optimizer)
        self.sample_signature = sample_signature
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

    def train_step(self, samples):
        """ train the model 1 step """
        # classifier
        with tf.GradientTape() as tape:
            # outputs of a forward run of model, potentially contains more than one item
            outputs = self.model(samples, training=True, stage="classifier")
            loss_c, metrics_c = self.model.get_loss(outputs, samples, training=True, stage="classifier")
            total_loss = sum(list(loss_c.values())) if isinstance(loss_c, dict) else loss_c
        grads = tape.gradient(total_loss, self.model.get_stage_model("classifier").trainable_variables)
        grads = self.clip_by_norm(grads, self.hparams.clip_norm)
        self.model.get_stage_model("classifier").optimizer.apply_gradients\
            (zip(grads, self.model.get_stage_model("classifier").trainable_variables))

        # generator
        with tf.GradientTape() as tape:
            # outputs of a forward run of model, potentially contains more than one item
            outputs = self.model(samples, training=True, stage="generator")
            loss_g, metrics_g = self.model.get_loss(outputs, samples, training=True, stage="generator")
            total_loss = sum(list(loss_g.values())) if isinstance(loss_g, dict) else loss_g
        grads = tape.gradient(total_loss, self.model.get_stage_model("generator").trainable_variables)
        grads = self.clip_by_norm(grads, self.hparams.clip_norm)
        self.model.get_stage_model("generator").optimizer.apply_gradients\
            (zip(grads, self.model.get_stage_model("generator").trainable_variables))

        # discriminator
        with tf.GradientTape() as tape:
            # outputs of a forward run of model, potentially contains more than one item
            outputs = self.model(samples, training=True, stage="discriminator")
            loss_d, metrics_d = self.model.get_loss(outputs, samples, training=True, stage="discriminator")
            total_loss = sum(list(loss_d.values())) if isinstance(loss_d, dict) else loss_d
        grads = tape.gradient(total_loss, self.model.get_stage_model("discriminator").trainable_variables)
        grads = self.clip_by_norm(grads, self.hparams.clip_norm)
        self.model.get_stage_model("discriminator").optimizer.apply_gradients\
            (zip(grads, self.model.get_stage_model("discriminator").trainable_variables))

        # total loss is the same of classifier, generator and discriminator
        total_loss = loss_c + loss_g + loss_d
        metrics = {"metrics_c": metrics_c, "metrics_g": metrics_g, "metrics_d": metrics_d}
        return total_loss, metrics

    def train(self, dataset, total_batches=-1):
        """ Update the model in 1 epoch """
        train_step = self.train_step
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            train_step = tf.function(train_step, input_signature=self.sample_signature)
        for batch, samples in enumerate(dataset.take(total_batches)):
            # train 1 step
            samples = self.model.prepare_samples(samples)
            total_loss, metrics = train_step(samples)
            if batch % self.hparams.log_interval == 0:
                logging.info(self.metric_checker(total_loss, metrics))
                self.model.reset_metrics()

    def evaluate_step(self, samples):
        """ evaluate the model 1 step """
        # outputs of a forward run of model, potentially contains more than one item
        outputs = self.model(samples, training=False, stage="classifier")
        loss_c, metrics_c = self.model.get_loss(outputs, samples, training=False, stage="classifier")

        outputs = self.model(samples, training=False, stage="generator")
        loss_g, metrics_g = self.model.get_loss(outputs, samples, training=False, stage="generator")

        outputs = self.model(samples, training=False, stage="discriminator")
        loss_d, metrics_d = self.model.get_loss(outputs, samples, training=False, stage="discriminator")

        total_loss = loss_c + loss_g + loss_d
        metrics = {"metrics_c": metrics_c, "metrics_g": metrics_g, "metrics_d": metrics_d}
        return total_loss, metrics

    def evaluate(self, dataset, epoch=0):
        """ evaluate the model """
        loss_metric = tf.keras.metrics.Mean(name="AverageLoss")
        loss, metrics = None, None
        evaluate_step = self.evaluate_step
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            evaluate_step = tf.function(evaluate_step, input_signature=self.sample_signature)
        self.model.reset_metrics()  # init metric.result() with 0
        for batch, samples in enumerate(dataset):
            samples = self.model.prepare_samples(samples)
            total_loss, metrics = evaluate_step(samples)
            if batch % self.hparams.log_interval == 0:
                logging.info(self.metric_checker(total_loss, metrics, -2))
            total_loss = sum(list(total_loss.values())) if isinstance(total_loss, dict) else total_loss
            loss_metric.update_state(total_loss)
        logging.info(self.metric_checker(loss_metric.result(), metrics, evaluate_epoch=epoch))
        self.model.reset_metrics()
        return loss_metric.result(), metrics


class ConvertSolver(BaseSolver):
    """ ConvertSolver
    """
    default_config = {
        "output_directory": "./gen_vcc2018/",
        "model_avg_num": 1,
        "fs": 22050
    }

    def __init__(self, model=None, data_descriptions=None, config=None):
        super().__init__(model, None, None)
        self.model = model
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.feature_normalizer = data_descriptions.feature_normalizer
        self.speakers_ids_dict = data_descriptions.speakers_ids_dict
        self.fs, self.fft_size = data_descriptions.fs, data_descriptions.fft_size

    def inference(self, dataset, rank_size=1):
        if dataset is None:
            print("convert dataset error!")
            return
        for i, samples in enumerate(dataset):
            samples = self.model.prepare_samples(samples)
            src_coded_sp, src_speaker_onehot, src_f0, src_ap = samples["src_coded_sp"], \
                 samples["src_speaker"], samples["src_f0"], samples["src_ap"]
            tar_speaker_onehot = samples["tar_speaker"]
            # Map the ids to the speakers name
            src_id, tar_id = samples["src_id"], samples["tar_id"]
            src_id, tar_id = int(src_id), int(tar_id)
            src_speaker = self.speakers_ids_dict[src_id]
            tar_speaker = self.speakers_ids_dict[tar_id]

            src_wav_filename = samples["src_wav_filename"]
            src_filename = src_wav_filename.numpy()[0].decode().replace(".npz","")

            gen_coded_sp = self.model.convert(src_coded_sp, tar_speaker_onehot)
            gen_coded_sp = tf.transpose(tf.squeeze(gen_coded_sp), [1, 0])
            coded_sp = self.feature_normalizer(gen_coded_sp, str(tar_speaker), reverse=True)

            def apply_f0_cmvn(cmvn_dict, feat_data, src_speaker, tar_speaker):
                if tar_speaker not in cmvn_dict:
                    print("tar_speaker not in cmvn_dict!")
                    return feat_data
                f0 = feat_data.numpy()
                src_mean = cmvn_dict[src_speaker][2]
                src_var = cmvn_dict[src_speaker][3]
                tar_mean = cmvn_dict[tar_speaker][2]
                tar_var = cmvn_dict[tar_speaker][3]
                f0_converted = np.exp((np.ma.log(f0) - src_mean) / np.sqrt(src_var) * np.sqrt(tar_var) + tar_mean)
                return f0_converted
            f0 = apply_f0_cmvn(self.feature_normalizer.cmvn_dict, src_f0, str(src_speaker), str(tar_speaker))

            # Restoration of sp characteristics
            c = []
            for one_slice in coded_sp:
                one_slice = np.ascontiguousarray(one_slice, dtype=np.float64).reshape(1, -1)
                decoded_sp = pyworld.decode_spectral_envelope(one_slice, self.fs, fft_size=self.fft_size)
                c.append(decoded_sp)
            sp = np.concatenate((c), axis=0)
            f0 = np.squeeze(f0, axis=(0,)).astype(np.float64)
            src_ap = np.squeeze(src_ap.numpy(), axis=(0,)).astype(np.float64)

            # Remove the extra padding at the end of the sp feature
            sp = sp[:src_ap.shape[0], :]
            # sp: T,fft_size//2+1   f0: T   ap: T,fft_size//2+1
            synwav = pyworld.synthesize(f0, sp, src_ap, self.fs)

            wavname = src_speaker + "_" + tar_speaker + "_" + src_filename  + ".wav"
            wavfolder = os.path.join(self.hparams.output_directory)
            if not os.path.exists(wavfolder):
                os.makedirs(wavfolder)
            wavpath = os.path.join(wavfolder, wavname)

            librosa.output.write_wav(wavpath, synwav, sr=self.fs)
            print("generate wav:", wavpath)

