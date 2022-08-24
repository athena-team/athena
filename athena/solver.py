# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Jianwei Sun; Ruixiong Zhang; Dongwei Jiang; Chunxin Xiao; Chaoyang Mei
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

from .utils.hparam import register_and_parse_hparams
from .utils.metric_check import MetricChecker
from .utils.misc import validate_seqs
from .metrics import CharactorAccuracy

from .models.lm.kenlm import NgramLM
from .tools.vocoder import GriffinLim
import math


class BaseSolver(tf.keras.Model):
    """Base Training Solver.
    """
    default_config = {
        "clip_norm": 100.0,
        "log_interval": 10,
        "ckpt_interval_train": 10000,
        "ckpt_interval_dev": 10000,
        "enable_tf_function": True,
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

    def train(self, trainset, devset, checkpointer, pbar, epoch, total_batches=-1):
        """ Update the model in 1 epoch """
        train_step = self.train_step
        pbar.set_description('Training ')  # it=iterations
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            train_step = tf.function(train_step, input_signature=self.sample_signature)
        for batch, samples in enumerate(trainset.take(total_batches)):
            # train 1 step
            samples = self.model.prepare_samples(samples)
            loss, metrics = train_step(samples)
            if (batch+1) % self.hparams.log_interval == 0:
                pbar.update(self.hparams.log_interval)
                logging.info(self.metric_checker(loss, metrics))
                self.model.reset_metrics()
            if (batch+1) % self.hparams.ckpt_interval_train == 0:
                checkpointer(loss, metrics, training=True)
            if (batch+1) % self.hparams.ckpt_interval_dev == 0:
                dev_loss, dev_metrics = self.evaluate(devset, epoch)
                checkpointer(dev_loss, dev_metrics, training=False)

    def save_checkpointer(self, checkpointer, devset, epoch):
        if self.hparams.do_dev_test_interval:
            loss, metrics = self.evaluate(devset, epoch)
        checkpointer(loss, metrics)

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
        if os.getenv('HOROVOD_TRAIN_MODE', 'normal') == 'fast':
            tape = hvd.DistributedGradientTape(tape, sparse_as_dense=True, op=hvd.Adasum)
        else:
            tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads = self.clip_by_norm(grads, self.hparams.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, metrics

    def train(self, trainset, devset, checkpointer, pbar, epoch, total_batches=-1):
        """ Update the model in 1 epoch """
        train_step = self.train_step
        pbar.set_description('Processing ')  # it=iterations
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            train_step = tf.function(train_step, input_signature=self.sample_signature)
        for batch, samples in enumerate(trainset.take(total_batches)):
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
            if hvd.rank() == 0:
                if (batch+1) % self.hparams.log_interval == 0:
                    pbar.update(self.hparams.log_interval)
                    logging.info(self.metric_checker(loss, metrics))
                    self.model.reset_metrics()

                if (batch + 1) % self.hparams.ckpt_interval_train == 0:
                    checkpointer(loss, metrics, training=True)
                if (batch + 1) % self.hparams.ckpt_interval_dev == 0:
                    dev_loss, dev_metrics = self.evaluate(devset, epoch)
                    checkpointer(dev_loss, dev_metrics, training=False)

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
    """ ASR DecoderSolver
    """
    default_config = {
        "inference_type": "asr",
        "decoder_type": "wfst_decoder",
        "enable_tf_function": False,
        "model_avg_num": 1,
        "sort_by": "CTCAccuracy",
        "sort_by_time": False,  # ignore "sort_by", if sort_by_time is True.
        "sort_reverse": True,
        "beam_size": 4,
        "pre_beam_size": 30,
        "at_weight": 1.0,
        "ctc_weight": 0.0,
        "simulate_streaming": False,
        "decoding_chunk_size": -1,
        "num_decoding_left_chunks": -1,
        "lm_weight": 0.1,
        "lm_type": None,
        "lm_path": None,
        "lm_nbest_avg": None,
        "predict_path": None,
        "label_path": None,
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
        self.lm_model = None
        if self.hparams.lm_path is not None:
            if self.hparams.lm_type is not None and self.hparams.lm_type in ["rnn", "transformer"]:
                from athena.main import build_model_from_jsonfile
                _, lm_checkpointer, _ = build_model_from_jsonfile(self.hparams.lm_path)
                if self.hparams.lm_nbest_avg:
                    lm_checkpointer.compute_nbest_avg(int(self.hparams.lm_nbest_avg), reverse=False)
                else:
                    lm_checkpointer.restore_from_best()
                self.lm_model = lm_checkpointer.model
            elif self.hparams.lm_type is not None and self.hparams.lm_type in ["ngram"]:
                self.lm_model = NgramLM(self.hparams.lm_path)
            else:
                raise NotImplementedError("'lm_model' only support rnn, transformer and ngram, please appoint it!")

    def inference(self, dataset_builder, rank_size=1, conf=None):
        """ decode the model """
        if dataset_builder is None:
            return
        dataset = dataset_builder.as_dataset(batch_size=conf.batch_size)
        metric = CharactorAccuracy(rank_size=rank_size)
        st = time.time()
        if self.hparams.predict_path is None or self.hparams.label_path is None:
            logging.warning("Please set inference_config.predict_path and inference_config.label_path!")
            self.hparams.predict_path = "inference.result"
            self.hparams.label_path = "inference.label"
        if self.lm_model is not None:
            self.lm_model.text_featurizer = dataset_builder.text_featurizer
        # self.lm_model.ignored_id = [self.model.model.eos, self.model.model.sos]
        if not os.path.exists(os.path.dirname(self.hparams.predict_path)):
            os.mkdir(os.path.dirname(self.hparams.predict_path))

        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            self.model.enable_tf_funtion()

        with open(self.hparams.predict_path, "w") as predict_file_out:
            with open(self.hparams.label_path, "w") as label_file_out:
                for _, samples in enumerate(dataset):
                    begin = time.time()
                    samples = self.model.prepare_samples(samples)
                    predictions = self.model.decode(samples, self.hparams, self.lm_model)

                    # transform (tensor of ids) to (list of char or symbol)
                    predict_txt_list = dataset_builder.text_featurizer. \
                        decode_to_list(predictions.numpy().tolist(), [self.model.model.eos, self.model.model.sos])
                    for predict, utt_id in zip(predict_txt_list, samples["utt_id"]):
                        predict_file_out.write(" ".join(predict) + "({utt_id})\n".format(utt_id=utt_id.numpy().decode()))

                    label_txt_list = dataset_builder.text_featurizer. \
                        decode_to_list(samples["output"].numpy().tolist(), [self.model.model.eos, self.model.model.sos])
                    for label, utt_id in zip(label_txt_list, samples["utt_id"]):
                        label_file_out.write(" ".join(label) + "({utt_id})\n".format(utt_id=utt_id.numpy().decode()))
                    predict_file_out.flush(), label_file_out.flush()

                    predictions = tf.cast(predictions, tf.int64)
                    validated_preds, _ = validate_seqs(predictions, self.model.eos)
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
                ed = time.time()
                logging.info("decoding finished, cost %.4f s" % (ed - st))

    def inference_saved_model(self, dataset_builder, rank_size=1, conf=None):
        """ decode the model """
        if dataset_builder is None:
            return
        dataset = dataset_builder.as_dataset(batch_size=conf.batch_size)
        metric = CharactorAccuracy(rank_size=rank_size)
        st = time.time()
        if self.hparams.predict_path is None or self.hparams.label_path is None:
            logging.warning("Please set inference_config.predict_path and inference_config.label_path!")
            self.hparams.predict_path = "inference.result"
            self.hparams.label_path = "inference.label"
        if self.lm_model is not None:
            self.lm_model.text_featurizer = dataset_builder.text_featurizer
        # self.lm_model.ignored_id = [self.model.model.eos, self.model.model.sos]
        if not os.path.exists(os.path.dirname(self.hparams.predict_path)):
            os.mkdir(os.path.dirname(self.hparams.predict_path))
        with open(self.hparams.predict_path, "w") as predict_file_out:
            with open(self.hparams.label_path, "w") as label_file_out:
                for _, samples in enumerate(dataset):
                    begin = time.time()
                    #samples = self.model.prepare_samples(samples)
                    predictions = self.model.deploy_function(samples['input'])

                    # transform (tensor of ids) to (list of char or symbol)
                    predict_txt_list = dataset_builder.text_featurizer. \
                        decode_to_list(predictions.numpy().tolist(), [dataset_builder.num_class, dataset_builder.num_class])
                    for predict, utt_id in zip(predict_txt_list, samples["utt_id"]):
                        predict_file_out.write(" ".join(predict) + "({utt_id})\n".format(utt_id=utt_id.numpy().decode()))

                    label_txt_list = dataset_builder.text_featurizer. \
                        decode_to_list(samples["output"].numpy().tolist(), [dataset_builder.num_class, dataset_builder.num_class])
                    for label, utt_id in zip(label_txt_list, samples["utt_id"]):
                        label_file_out.write(" ".join(label) + "({utt_id})\n".format(utt_id=utt_id.numpy().decode()))
                    predict_file_out.flush(), label_file_out.flush()

                    predictions = tf.cast(predictions, tf.int64)
                    validated_preds, _ = validate_seqs(predictions, dataset_builder.num_class)
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
                ed = time.time()
                logging.info("decoding finished, cost %.4f s" % (ed - st))

class SynthesisSolver(BaseSolver):
    """ SynthesisSolver (TTS Solver)
    """
    default_config = {
        "clip_norm": 100.0,
        "log_interval": 10,
        "ckpt_interval_train": 10000,
        "ckpt_interval_dev": 10000,
        "enable_tf_function": True,
        "model_avg_num": 1,
        "gl_iters": 64,
        "synthesize_from_true_fbank": True,
        "output_directory": None
    }

    def __init__(self, model, optimizer=None, sample_signature=None, eval_sample_signature=None,
                 config=None, **kwargs):
        super().__init__(model, optimizer, sample_signature)
        self.optimizer = optimizer
        self.metric_checker = MetricChecker(self.optimizer)
        self.eval_sample_signature = eval_sample_signature
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

    def inference(self, dataset_builder, rank_size=1, conf=None):
        """ synthesize using vocoder on dataset """
        if dataset_builder is None:
            return
        feature_normalizer = dataset_builder.feature_normalizer
        speakers_ids_dict = dataset_builder.speakers_ids_dict
        vocoder = GriffinLim(dataset_builder)
        # dataset = dataset_builder.as_dataset(batch_size=conf.batch_size)
        dataset = dataset_builder.as_dataset(batch_size=1)

        total_elapsed = 0
        total_seconds = 0
        synthesize_step = tf.function(self.model.synthesize, input_signature=self.sample_signature)
        for i, samples in enumerate(dataset):
            samples = self.model.prepare_samples(samples)
            speaker = samples['speaker']
            speaker = speakers_ids_dict[int(speaker[0])]
            start = time.time()
            outputs = synthesize_step(samples)
            end = time.time() - start
            total_elapsed += end
            features, _ = outputs
            utt_id = samples['utt_id'].numpy()[0].decode()
            if self.hparams.synthesize_from_true_fbank:
                samples_outputs = feature_normalizer(samples['output'][0],
                                                     speaker, reverse=True)
                vocoder(samples_outputs.numpy(), self.hparams, name='true_%s' % utt_id)
            features = feature_normalizer(features[0], speaker, reverse=True)
            seconds = vocoder(features.numpy(), self.hparams, name=utt_id)
            total_seconds += seconds
        logging.info("model computation elapsed: %s\ttotal seconds: %s\tRTF: %.4f"
                     % (total_elapsed, total_seconds, float(total_elapsed / total_seconds)))

    def inference_saved_model(self, dataset_builder, rank_size=1, conf=None):
        """ synthesize using vocoder on dataset """
        if dataset_builder is None:
            return
        feature_normalizer = dataset_builder.feature_normalizer
        speakers_ids_dict = dataset_builder.speakers_ids_dict
        vocoder = GriffinLim(dataset_builder)
        dataset = dataset_builder.as_dataset(batch_size=1)

        total_elapsed = 0
        total_seconds = 0
        synthesize_step = self.model.deploy_function

        for i, samples in enumerate(dataset):
            speaker = samples['speaker']
            speaker = speakers_ids_dict[int(speaker[0])]
            start = time.time()
            outputs = synthesize_step(samples['input'])
            end = time.time() - start
            total_elapsed += end
            features = outputs
            utt_id = samples['utt_id'].numpy()[0].decode()
            if self.hparams.synthesize_from_true_fbank:
                samples_outputs = feature_normalizer(samples['output'][0],
                                                     speaker, reverse=True)
                vocoder(samples_outputs.numpy(), self.hparams, name='true_%s' % utt_id)
            features = feature_normalizer(features[0], speaker, reverse=True)
            seconds = vocoder(features.numpy(), self.hparams, name=utt_id)
            total_seconds += seconds
        logging.info("model computation elapsed: %s\ttotal seconds: %s\tRTF: %.4f"
                     % (total_elapsed, total_seconds, float(total_elapsed / total_seconds)))

class VadSolver(BaseSolver):
    """ VadSolver
    """
    default_config = {
        "inference_type": "vad",
        "model_avg_num": 1,
        "clip_norm": 100.0,
        "log_interval": 10,
        "ckpt_interval_train": 10000,
        "ckpt_interval_dev": 10000,
        "enable_tf_function": True,
        "video_skip": 4
    }

    # pylint: disable=super-init-not-called
    def __init__(self, model, optimizer=None, sample_signature=None, eval_sample_signature=None, data_descriptions=None, config=None):
        super().__init__(model, None, None)
        self.model = model
        self.optimizer = optimizer
        self.metric_checker = MetricChecker(self.optimizer)
        self.sample_signature = sample_signature
        self.eval_sample_signature = eval_sample_signature
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

    def inference(self, dataset, rank_size=1, conf=None):
        """ decode the model """
        if dataset is None:
            return
        acc_metric = tf.keras.metrics.Accuracy(name="Accuracy")
        recall_metric = tf.keras.metrics.Recall(name="Recall")
        for _, samples in enumerate(dataset):
            samples = self.model.prepare_samples(samples)
            outputs = self.model(samples, training=False)
            outputs_argmax = tf.argmax(outputs, axis=2, output_type=tf.int32)
            samples_outputs = tf.reshape(samples["output"], (1, -1))
            outputs_argmax = tf.reshape(outputs_argmax, (1, -1))
            print("utt:{}\tlabels:{}\tpredict:{}".format(samples["utt"], samples_outputs, outputs_argmax))
            acc_metric.update_state(samples_outputs, outputs_argmax)
            recall_metric.update_state(samples_outputs, outputs_argmax)
        reports = (
            "Accuracy: %.4f\tRecall: %.4f"
            % (
                acc_metric.result(),
                recall_metric.result(),
            )
        )
        logging.info(reports)
        logging.info("vad test finished")


"""
Audio and video task solver
"""

class AVSolver(tf.keras.Model):
    """Base Solver.
    """
    default_config = {
        "clip_norm": 100.0,
        "log_interval": 10,
        "ckpt_interval_train": 10000,
        "ckpt_interval_dev": 10000,
        "enable_tf_function": True,
        "video_skip": 4
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
            if self.model.name !='mtl_transformer_ctc':
                outputs = self.model(samples, training=True)
            else:
                outputs = self.model(samples, training=True)
            loss, metrics = self.model.get_loss(outputs, samples, training=True)
            total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads = self.clip_by_norm(grads, self.hparams.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, metrics

    def train(self, trainset, devset, checkpointer, pbar, epoch, total_batches=-1):
        """ Update the model in 1 epoch """
        train_step = self.train_step
        pbar.set_description('Training ')  # it=iterations
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            train_step = tf.function(train_step, input_signature=self.sample_signature)
        for batch, samples in enumerate(trainset.take(total_batches)):
            # train 1 step
            samples = self.model.prepare_samples(samples)
            frame = math.ceil(samples["input"].shape[1] / self.hparams.video_skip)
            if frame + 1 > samples["video"].shape[1]:
                frame = samples["video"].shape[1]
                audio_frame = frame * self.hparams.video_skip
                audio_frames = tf.ones(samples["input_length"].shape,dtype=tf.dtypes.int32)*audio_frame
                samples["input"] = samples["input"][:, :audio_frame, :]
                samples["video"] = samples["video"][:, :frame, :]
                samples["input_length"] = tf.where(samples["input_length"]>audio_frames,audio_frames,samples["input_length"])
            else:
                samples["video"] = samples["video"][:, 1:frame + 1, :]
            loss, metrics = train_step(samples)
            if (batch+1) % self.hparams.log_interval == 0:
                pbar.update(self.hparams.log_interval)
                logging.info(self.metric_checker(loss, metrics))
                self.model.reset_metrics()
            if (batch+1) % self.hparams.ckpt_interval_train == 0:
                checkpointer(loss, metrics, training=True)
            if (batch+1) % self.hparams.ckpt_interval_dev == 0:
                dev_loss, dev_metrics = self.evaluate(devset, epoch)
                checkpointer(dev_loss, dev_metrics, training=False)



    def evaluate_step(self, samples):
        """ evaluate the model 1 step """
        # outputs of a forward run of model, potentially contains more than one item
        if self.model.name != 'mtl_transformer_ctc':
            outputs = self.model(samples, training=False)
        else:
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
            frame = math.ceil(samples["input"].shape[1]/self.hparams.video_skip)
            if frame + 1 > samples["video"].shape[1]:
                frame = samples["video"].shape[1]
                audio_frame = frame * self.hparams.video_skip
                audio_frames = tf.ones(samples["input_length"].shape,dtype=tf.dtypes.int32)*audio_frame
                samples["input"] = samples["input"][:, :audio_frame, :]
                samples["video"] = samples["video"][:, :frame, :]
                samples["input_length"] = tf.where(samples["input_length"]>audio_frames,audio_frames,samples["input_length"])
            else:
                samples["video"] = samples["video"][:, 1:frame + 1, :]
            loss, metrics = evaluate_step(samples)
            if batch % self.hparams.log_interval == 0:
                logging.info(self.metric_checker(loss, metrics, -2))
            total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
            loss_metric.update_state(total_loss)
        logging.info(self.metric_checker(loss_metric.result(), metrics, evaluate_epoch=epoch))
        self.model.reset_metrics()
        return loss_metric.result(), metrics


class AVHorovodSolver(AVSolver):
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
            if self.model.name !='mtl_transformer_ctc':
                outputs = self.model(samples, training=True)
            else:
                outputs = self.model(samples, training=True)
            loss, metrics = self.model.get_loss(outputs, samples, training=True)
            total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
        # Horovod: add Horovod Distributed GradientTape.
        if os.getenv('HOROVOD_TRAIN_MODE', 'normal') == 'fast':
            tape = hvd.DistributedGradientTape(tape, sparse_as_dense=True, op=hvd.Adasum)
        else:
            tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        grads = self.clip_by_norm(grads, self.hparams.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, metrics


    def train(self, trainset, devset, checkpointer, pbar, epoch, total_batches=-1):
        """ Update the model in 1 epoch """
        train_step = self.train_step
        pbar.set_description('Processing ')  # it=iterations
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            train_step = tf.function(train_step, input_signature=self.sample_signature)
        for batch, samples in enumerate(trainset.take(total_batches)):
            # train 1 step
            samples = self.model.prepare_samples(samples)
            frame = math.ceil(samples["input"].shape[1]/self.hparams.video_skip)
            if frame + 1 > samples["video"].shape[1]:
                print("video is shorter than audio")
                frame = samples["video"].shape[1]
                audio_frame = frame * self.hparams.video_skip
                audio_frames = tf.ones(samples["input_length"].shape,dtype=tf.dtypes.int32)*audio_frame
                samples["input"] = samples["input"][:, :audio_frame, :]
                samples["video"] = samples["video"][:, :frame, :]
                samples["input_length"] = tf.where(samples["input_length"]>audio_frames,audio_frames,samples["input_length"])
            else:
                samples["video"] = samples["video"][:, 1:frame + 1, :]
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
            if hvd.rank() == 0:
                if (batch+1) % self.hparams.log_interval == 0:
                    pbar.update(self.hparams.log_interval)
                    logging.info(self.metric_checker(loss, metrics))
                    self.model.reset_metrics()

                if (batch + 1) % self.hparams.ckpt_interval_train == 0:
                    checkpointer(loss, metrics, training=True)
                if (batch + 1) % self.hparams.ckpt_interval_dev == 0:
                    dev_loss, dev_metrics = self.evaluate(devset, epoch)
                    checkpointer(dev_loss, dev_metrics, training=False)


    def evaluate(self, dataset, epoch=0):
        """ evaluate the model """
        loss_metric = tf.keras.metrics.Mean(name="AverageLoss")
        loss, metrics = None, None
        evaluate_step = self.evaluate_step
        if self.hparams.enable_tf_function:
            logging.info("please be patient, enable tf.function, it takes time ...")
            evaluate_step = tf.function(evaluate_step, input_signature=self.sample_signature)
        self.model.reset_metrics()
        for batch, samples in enumerate(dataset):
            samples = self.model.prepare_samples(samples)
            frame = math.ceil(samples["input"].shape[1]/self.hparams.video_skip)
            if frame + 1 > samples["video"].shape[1]:
                print("video is shorter than audio")
                frame = samples["video"].shape[1]
                audio_frame = frame * self.hparams.video_skip
                audio_frames = tf.ones(samples["input_length"].shape,dtype=tf.dtypes.int32)*audio_frame
                samples["input"] = samples["input"][:, :audio_frame, :]
                samples["video"] = samples["video"][:, :frame, :]
                samples["input_length"] = tf.where(samples["input_length"]>audio_frames,audio_frames,samples["input_length"])
            else:
                samples["video"] = samples["video"][:, 1:frame + 1, :]
            loss, metrics = evaluate_step(samples)
            if batch % self.hparams.log_interval == 0 and hvd.rank() == 0:
                logging.info(self.metric_checker(loss, metrics, -2))
            total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
            loss_metric.update_state(total_loss)
        if hvd.rank() == 0:
            logging.info(self.metric_checker(loss_metric.result(), metrics, evaluate_epoch=epoch))
            self.model.reset_metrics()
        return loss_metric.result(), metrics

class AVDecoderSolver(AVSolver):
    """ DecoderSolver
    """
    default_config = {
        "inference_type": "asr",
        "decoder_type": "wfst_decoder",
        "model_avg_num": 1,
        "beam_size": 4,
        "pre_beam_size": 30,
        "at_weight": 1.0,
        "ctc_weight": 0.0,
        "lm_weight": 0.1,
        "lm_type": "",
        "lm_path": None,
        "predict_path": None,
        "label_path": None,
        "acoustic_scale": 10.0,
        "max_active": 80,
        "min_active": 0,
        "wfst_beam": 30.0,
        "max_seq_len": 100,
        "video_skip": 4,
        "wfst_graph": None
    }

    # pylint: disable=super-init-not-called
    def __init__(self, model, data_descriptions=None, config=None):
        super().__init__(model, None, None)
        self.model = model
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.lm_model = None
        if self.hparams.lm_type == "rnn":
            from athena.main import build_model_from_jsonfile
            _, self.lm_model, _, lm_checkpointer, _ = build_model_from_jsonfile(self.hparams.lm_path)
            lm_checkpointer.restore_from_best()

    def inference(self, dataset_builder, rank_size=1, conf=None):
        """ decode the model """
        if dataset_builder is None:
            return
        dataset = dataset_builder.as_dataset(batch_size=conf.batch_size)
        metric = CharactorAccuracy(rank_size=rank_size)
        st = time.time()
        with open(self.hparams.predict_path, "w") as predict_file_out:
            with open(self.hparams.label_path, "w") as label_file_out:
                for _, samples in enumerate(dataset):
                    begin = time.time()
                    samples = self.model.prepare_samples(samples)
                    frame = math.ceil(samples["input"].shape[1] / self.hparams.video_skip)
                    if frame + 1 > samples["video"].shape[1]:
                        print("video is shorter than audio")
                        frame = samples["video"].shape[1]
                        audio_frame = frame * self.hparams.video_skip
                        audio_frames = tf.ones(samples["input_length"].shape, dtype=tf.dtypes.int32) * audio_frame
                        samples["input"] = samples["input"][:, :audio_frame, :]
                        samples["video"] = samples["video"][:, :frame, :]
                        samples["input_length"] = tf.where(samples["input_length"] > audio_frames, audio_frames,
                                                           samples["input_length"])
                    else:
                        samples["video"] = samples["video"][:, 1:frame + 1, :]

                    if self.hparams.decoder_type != "argmax_decoder":
                        output = self.model(samples,training=False)
                        predictions = self.model.decode(samples, self.hparams, self.lm_model)

                        # transform (tensor of ids) to (list of char or symbol)
                        predict_txt_list = dataset_builder.text_featurizer. \
                            decode_to_list(predictions.numpy().tolist(), [self.model.model.eos, self.model.model.sos])
                        for predict, utt in zip(predict_txt_list, samples["utt"]):
                            predict_file_out.write(" ".join(predict) + "({utt})\n".format(utt=utt.numpy().decode()))

                        label_txt_list = dataset_builder.text_featurizer. \
                            decode_to_list(samples["output"].numpy().tolist(), [self.model.model.eos, self.model.model.sos])
                        for label, utt in zip(label_txt_list, samples["utt"]):
                            label_file_out.write(" ".join(label) + "({utt})\n".format(utt=utt.numpy().decode()))
                        predict_file_out.flush(), label_file_out.flush()

                        predictions = tf.cast(predictions, tf.int64)
                        validated_preds, _ = validate_seqs(predictions, self.model.eos)
                        validated_preds = tf.cast(validated_preds, tf.int64)
                        num_errs, _ = metric.update_state(validated_preds, samples)

                        reports = (
                                "predictions: %s\tlabels: %s\terrs: %d\tavg_acc: %.4f\tsec/iter: %.4f\tkey: %s"
                                % (
                                    predictions,
                                    samples["output"].numpy(),
                                    num_errs,
                                    metric.result(),
                                    time.time() - begin,
                                    samples["utt"],
                                )
                        )
                        logging.info(reports)
                    else:
                        # predictions, score = self.model.decode(samples)
                        output = self.model(samples, training=False)
                        metrics = self.model.get_acc(output, samples, training=False)
                        print(samples['utt'])
                        print(metrics)
                ed = time.time()
                logging.info("decoding finished, cost %.4f s" % (ed - st))

    def inference_freeze(self, dataset_builder, rank_size=1, conf=None):
        """ decode the model """
        if dataset_builder is None:
            return
        dataset = dataset_builder.as_dataset(batch_size=conf.batch_size)
        st = time.time()
        with open(self.hparams.predict_path, "w") as predict_file_out:
            with open(self.hparams.label_path, "w") as label_file_out:
                for _, samples in enumerate(dataset):
                    frame = math.ceil(samples["input"].shape[1] / self.hparams.video_skip)
                    if frame + 1 > samples["video"].shape[1]:
                        print("video is shorter than audio")
                        frame = samples["video"].shape[1]
                        audio_frame = frame * self.hparams.video_skip
                        audio_frames = tf.ones(samples["input_length"].shape, dtype=tf.dtypes.int32) * audio_frame
                        samples["input"] = samples["input"][:, :audio_frame, :]
                        samples["video"] = samples["video"][:, :frame, :]
                        samples["input_length"] = tf.where(samples["input_length"] > audio_frames, audio_frames,
                                                           samples["input_length"])
                    else:
                        samples["video"] = samples["video"][:, 1:frame + 1, :]
                    begin = time.time()
                    softmax = tf.nn.softmax(self.model.deploy_function(samples['input']))
                    predictions = tf.math.argmax(softmax, axis=2)
                    label = samples["output"].numpy()[0].tolist()
                    label = [str(i) for i in label]
                    label = ' '.join(label)
                    predictions=predictions.numpy()[0].tolist()
                    predictions=[str(i) for i in predictions]
                    predictions=' '.join(predictions)
                    #metrics = self.model.get_acc(predictions, samples, training=False)
                    reports = (
                            "%s\t%s\t%s"
                            % (
                                samples["utt"].numpy()[0],
                                label,
                                predictions,

                            )
                    )
                    predict_file_out.write(str(samples["utt"].numpy()[0],'utf-8')+'\t' + predictions+'\n')
                    label_file_out.write(str(samples["utt"].numpy()[0],'utf-8')+'\t' + label+'\n')
                    logging.info(reports)
                ed = time.time()
                logging.info("decoding finished, cost %.4f s" % (ed - st))

    def inference_argmax(self, dataset_builder, rank_size=1, conf=None):
        """ decode the model """
        if dataset_builder is None:
            return
        dataset = dataset_builder.as_dataset(batch_size=conf.batch_size)
        st = time.time()
        with open(self.hparams.predict_path, "w") as predict_file_out:
            with open(self.hparams.label_path, "w") as label_file_out:
                for _, samples in enumerate(dataset):
                    frame = math.ceil(samples["input"].shape[1] / self.hparams.video_skip)
                    if frame + 1 > samples["video"].shape[1]:
                        print("video is shorter than audio")
                        frame = samples["video"].shape[1]
                        audio_frame = frame * self.hparams.video_skip
                        audio_frames = tf.ones(samples["input_length"].shape, dtype=tf.dtypes.int32) * audio_frame
                        samples["input"] = samples["input"][:, :audio_frame, :]
                        samples["video"] = samples["video"][:, :frame, :]
                        samples["input_length"] = tf.where(samples["input_length"] > audio_frames, audio_frames,
                                                           samples["input_length"])
                    else:
                        samples["video"] = samples["video"][:, 1:frame + 1, :]
                    begin = time.time()
                    softmax = tf.nn.softmax(self.model.decode(samples))
                    predictions = tf.math.argmax(softmax, axis=2)
                    label = samples["output"].numpy()[0].tolist()
                    label = [str(i) for i in label]
                    label = ' '.join(label)
                    predictions=predictions.numpy()[0].tolist()
                    predictions=[str(i) for i in predictions]
                    predictions=' '.join(predictions)
                    #metrics = self.model.get_acc(predictions, samples, training=False)
                    reports = (
                            "%s\t%s\t%s"
                            % (
                                samples["utt"].numpy()[0],
                                label,
                                predictions,

                            )
                    )
                    predict_file_out.write(str(samples["utt"].numpy()[0],'utf-8')+'\t' + predictions+'\n')
                    label_file_out.write(str(samples["utt"].numpy()[0],'utf-8')+'\t' + label+'\n')
                    logging.info(reports)
                ed = time.time()
                logging.info("decoding finished, cost %.4f s" % (ed - st))
