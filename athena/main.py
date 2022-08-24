# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Xiaoning Lei
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
# Only support tensorflow 2.0
# pylint: disable=invalid-name, no-member, wildcard-import, unused-wildcard-import, redefined-outer-name
""" starting point for model training """
import sys
import json
import tensorflow as tf
from absl import logging
from tqdm import tqdm, trange
from athena import *

SUPPORTED_DATASET_BUILDER = {
    #ASR dataset
    "speech_dataset": SpeechDatasetBuilder,
    "speech_recognition_dataset": SpeechRecognitionDatasetBuilder,
    "speech_recognition_dataset_batch_bins": SpeechRecognitionDatasetBatchBinsBuilder,
    "speech_recognition_dataset_kaldiio": SpeechRecognitionDatasetKaldiIOBuilder,
    "speech_recognition_dataset_batch_bins_kaldiio": SpeechRecognitionDatasetBatchBinsKaldiIOBuilder,
    #AV dataset for ASR
    "audio_videio_data_batch_bins": AudioVedioRecognitionDatasetBatchBinsBuilder,
    "audio_videio_data": AudioVedioRecognitionDatasetBuilder,
    # MPC dataset
    "mpc_speech_dataset": MpcSpeechDatasetBuilder,
    "mpc_speech_dataset_kaldiio": MpcSpeechDatasetKaldiIOBuilder,
    #LM dataset
    "language_dataset": LanguageDatasetBuilder,
    #VAD dataset
    "voice_activity_detection_dataset_kaldiio": VoiceActivityDetectionDatasetKaldiIOBuilder,
    #TTS dataset
    "speech_synthesis_dataset": SpeechSynthesisDatasetBuilder,
    "speech_fastspeech2_dataset": SpeechFastspeech2DatasetBuilder,
    #KWS dataset
    "speech_wakeup_dataset_kaldiio": SpeechWakeupDatasetKaldiIOBuilder,
    "speech_wakeup_framewise_dataset_kaldiio": SpeechWakeupFramewiseDatasetKaldiIOBuilder,
    "speech_wakeup_dataset_kaldiio_av": SpeechWakeupDatasetKaldiIOBuilderAVCE
}

SUPPORTED_MODEL = {
    #ASR model
    "mtl_transformer_ctc": MtlTransformerCtc,
    "mml_transformer": AV_MtlTransformer,
    "mpc": MaskedPredictCoding,
    #LM model
    "rnnlm": RNNLM,
    "transformer_lm": TransformerLM,
    #VAD mdoel
    "vad_dnn": VadDnn,
    "vad_marblenet": VadMarbleNet,
    #TTS model
    "tacotron2": Tacotron2,
    "fastspeech": FastSpeech,
    "fastspeech2": FastSpeech2,
    "tts_transformer": TTSTransformer,
    #KWS model
    "kws_dnn": DnnModel,
    "kws_cnn": CnnModel,
    "kws_crnn": CRnnModel,
    "kws_transformer": KWSTransformer,
    "kws_avtransformer": KWSAVTransformer,
    "kws_conformer": KWSConformer,
    "kws_transformer_focal_loss": KWSTransformer_FocalLoss
}

SUPPORTED_OPTIMIZER = {
    "warmup_adam": WarmUpAdam,
    "warmup_adam1": WarmUpAdam1,
    "expdecay_adam": ExponentialDecayAdam
}

SOLVERS = {
    "asr_base_solver": BaseSolver,
    "asr_decode_solver": DecoderSolver,
    "av_base_solver": AVSolver,
    "av_decode_solver": AVDecoderSolver,
    "tts_solver": SynthesisSolver,
    "vad_solver": VadSolver,
}

DEFAULT_CONFIGS = {
    "batch_size": 32,
    "num_epochs": 20,
    "sorta_epoch": 1,
    "ckpt": None,
    "test_ckpt": None,
    "ckpt_interval": 10000,
    "use_dev_loss": True,
    "summary_dir": None,
    "saved_model_dir":None,
    "log_dir": None,
    "solver_type": "asr_base_solver",
    "solver_gpu": [0],
    "solver_config": None,
    "model": "speech_transformer",
    "num_classes": None,
    "model_config": None,
    "pretrained_model": None,
    "load_tf_saved_model": None,
    "teacher_model": None,
    "optimizer": "warmup_adam",
    "optimizer_config": None,
    "convert_config": None,
    "num_data_threads": 1,
    "dataset_builder": "speech_recognition_dataset",
    "dev_dataset_builder": None,
    "trainset_config": None,
    "devset_config": None,
    "testset_config": None,
    "faset_config": None,
    "model_freeze": None,
    "model_optimization": None,
    "alignment_config": None,
    "inference_config": None
}


def parse_config(config):
    """ parse config """
    p = register_and_parse_hparams(DEFAULT_CONFIGS, config, cls="main")
    logging.info("hparams: {}".format(p))
    return p

def parse_jsonfile(jsonfile):
    """ parse the jsonfile, output the parameters
    """
    config = None
    with open(jsonfile) as file:
        config = json.load(file)
    p = register_and_parse_hparams(DEFAULT_CONFIGS, config, cls="main")
    logging.info("hparams: {}".format(p))
    return p

def build_model_from_jsonfile(jsonfile):
    """ creates model using configurations in json, load from checkpoint
    if previous models exist in checkpoint dir
    """
    p = parse_jsonfile(jsonfile)
    #if p.log_dir != None:
    #    logging.set_out_file(p.log_dir)
    if p.solver_type =="asr_decode_solver" or p.trainset_config == None:
        dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.testset_config)
        if p.testset_config is None and p.num_classes is None:
            raise ValueError("test_config and num_classes can not both be null")
    else:
        dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.trainset_config)
        if p.trainset_config is None and p.num_classes is None:
            raise ValueError("trainset_config and num_classes can not both be null")
    model = SUPPORTED_MODEL[p.model](
        data_descriptions=dataset_builder,
        config=p.model_config,
    )
    optimizer = SUPPORTED_OPTIMIZER[p.optimizer](p.optimizer_config)
    checkpointer = Checkpoint(
        checkpoint_directory=p.ckpt,
        use_dev_loss=p.use_dev_loss,
        model=model,
        optimizer=optimizer,
    )
    if p.pretrained_model is not None:
        # pre_run for lazy initilize in keras
        solver = SOLVERS[p.solver_type](
            model,
            optimizer,
            sample_signature=dataset_builder.sample_signature
        )
    return p, checkpointer, dataset_builder


def train(jsonfile, Solver, rank_size=1, rank=0):
    """
     entry point for model training, implements train loop
    :param jsonfile: json file to read configuration from
    :param Solver: an abstract class that implements high-level logic of train, evaluate, decode, etc
    :param rank_size: total number of workers, 1 if using single gpu
    :param rank: rank of current worker, 0 if using single gpu
    """
    p, checkpointer, trainset_builder = build_model_from_jsonfile(jsonfile)
    step = checkpointer.optimizer.iterations.numpy()
    # for cmvn
    trainset_builder.compute_cmvn_if_necessary(rank == 0)
    trainset_builder.shard(rank_size, rank)
    total_batch = int(trainset_builder.__len__()*p.num_epochs / p.batch_size // rank_size)
    epoch = int(step / (total_batch/p.num_epochs))
    if p.pretrained_model is not None and epoch == 0:
        p2, checkpointer2, _ = build_model_from_jsonfile(p.pretrained_model)
        checkpointer.model.restore_from_pretrained_model(checkpointer2.model, p2.model)
    if p.teacher_model is not None:
        p3, checkpointer3, _ = build_model_from_jsonfile(p.teacher_model)
        checkpointer.model.set_teacher_model(checkpointer3.model, p3.model)
    if rank == 0:
        set_default_summary_writer(p.summary_dir)

    if p.dev_dataset_builder is not None:
        devset_builder = p.dev_dataset_builder
    else:
        devset_builder = p.dataset_builder
    devset_builder = SUPPORTED_DATASET_BUILDER[devset_builder](p.devset_config)

    # train
    solver = Solver(
        checkpointer.model,
        checkpointer.optimizer,
        sample_signature=trainset_builder.sample_signature,
        eval_sample_signature=devset_builder.sample_signature,
        config=p.solver_config,
    )
    pbar = tqdm(total=int(total_batch))
    pbar.update(step)
    if epoch > p.num_epochs:
        logging.warning("current epoch %d exceeds than num_epochs %d" % (epoch, p.num_epochs))
        exit(0)
    while epoch < p.num_epochs:
        if rank == 0:
            logging.info(">>>>> start training in epoch %d" % epoch)
        if epoch >= p.sorta_epoch:
            trainset_builder.batch_wise_shuffle(batch_size=p.batch_size, epoch=epoch)
        solver.train(trainset_builder.as_dataset(p.batch_size, p.num_data_threads),
                     devset_builder.as_dataset(p.batch_size, p.num_data_threads),
                     checkpointer, pbar, epoch)

        if rank == 0:
            logging.info(">>>>> start evaluate in epoch %d" % epoch)
        devset = devset_builder.as_dataset(p.batch_size, p.num_data_threads)
        loss, metrics = solver.evaluate(devset, epoch)

        if rank == 0:
            checkpointer(loss, metrics, training=False)

        epoch = epoch + 1


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)

    jsonfile = sys.argv[1]
    p = parse_jsonfile(jsonfile)
    solver = SOLVERS[p.solver_type]
    solver.initialize_devices(p.solver_gpu)
    train(jsonfile, solver, 1, 0)
