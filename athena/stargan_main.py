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
""" entry point for stargan_vc model training """
import sys
import json
import tensorflow as tf
from absl import logging
from athena import *

SUPPORTED_DATASET_BUILDER = {
    "speech_recognition_dataset": SpeechRecognitionDatasetBuilder,
    "speech_recognition_dataset_kaldiio": SpeechRecognitionDatasetKaldiIOBuilder,
    "speech_systhesis_dataset": SpeechSynthesisDatasetBuilder,
    "speech_dataset": SpeechDatasetBuilder,
    "speech_dataset_kaldiio": SpeechDatasetKaldiIOBuilder,
    "language_dataset": LanguageDatasetBuilder,
    "voice_conversion_dataset": VoiceConversionDatasetBuilder
}

SUPPORTED_MODEL = {
    "deep_speech": DeepSpeechModel,
    "speech_transformer": SpeechTransformer,
    "speech_transformer2": SpeechTransformer2,
    "mtl_transformer_ctc": MtlTransformerCtc,
    "mpc": MaskedPredictCoding,
    "rnnlm": RNNLM,
    "translate_transformer": NeuralTranslateTransformer,
    "tacotron2": Tacotron2,
    "stargan" : StarganModel
}

SUPPORTED_OPTIMIZER = {
    "warmup_adam": WarmUpAdam,
    "expdecay_adam": ExponentialDecayAdam
}

DEFAULT_CONFIGS = {
    "batch_size": 32,
    "num_epochs": 20,
    "sorta_epoch": 1,
    "ckpt": None,
    "summary_dir": None,
    "solver_type": "vc",
    "solver_gpu": [0],
    "solver_config": None,
    "model": "speech_transformer",
    "num_classes": None,
    "model_config": None,
    "pretrained_model": None,
    "optimizer": "warmup_adam",
    "optimizer_config": None,
    "convert_config": None,
    "num_data_threads": 1,
    "dataset_builder": "speech_recognition_dataset",
    "trainset_config": None,
    "devset_config": None,
    "testset_config": None,
    "inference_config": None
}

def parse_config(config):
    """ parse config """
    p = register_and_parse_hparams(DEFAULT_CONFIGS, config, cls="main")
    logging.info("hparams: {}".format(p))
    return p

def build_model_from_jsonfile_stargan(jsonfile, pre_run=True):
    """ creates model using configurations in json, load from checkpoint
    if previous models exist in checkpoint dir.
    This function differs from build_model_from_jsonfile because it uses multiple 
    optimizer for stargan model
    """
    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)

    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.trainset_config)
    if p.trainset_config is None and p.num_classes is None:
        raise ValueError("trainset_config and num_classes can not both be null")
    model = SUPPORTED_MODEL[p.model](
        data_descriptions=dataset_builder,
        config=p.model_config,
    )
    checkpointer = Checkpoint(
        checkpoint_directory=p.ckpt,
        model=model,
        optimizer_g=model.get_stage_model("generator").optimizer,
	optimizer_d=model.get_stage_model("discriminator").optimizer,
	optimizer_c=model.get_stage_model("classifier").optimizer
    )
    
    if pre_run or p.pretrained_model is not None:
        # pre_run for lazy initialize in keras
        solver = GanSolver(
            model,
            sample_signature=dataset_builder.sample_signature
        )
        dataset = dataset_builder.as_dataset(p.batch_size)
        solver.evaluate_step(model.prepare_samples(iter(dataset).next()))
    return p, model, checkpointer


def train(jsonfile, Solver, rank_size=1, rank=0):
    """ entry point for model training, implements train loop
	:param jsonfile: json file to read configuration from
	:param Solver: an abstract class that implements high-level logic of train, evaluate, decode, etc
	:param rank_size: total number of workers, 1 if using single gpu
	:param rank: rank of current worker, 0 if using single gpu
	"""
    p, model, checkpointer = build_model_from_jsonfile_stargan(jsonfile)
    epoch = int(checkpointer.save_counter)
    if p.pretrained_model is not None and epoch == 0:
        p2, pretrained_model, _, _ = build_model_from_jsonfile_stargan(p.pretrained_model)
        model.restore_from_pretrained_model(pretrained_model, p2.model)

    if rank == 0:
        set_default_summary_writer(p.summary_dir)

    # for cmvn
    trainset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.trainset_config)
    trainset_builder.compute_cmvn_if_necessary(rank == 0)
    trainset_builder.shard(rank_size, rank)

    solver = Solver(
        model,
        sample_signature=trainset_builder.sample_signature,
        config=p.solver_config,
    )
    loss_g = 0
    metrics_g = {}
    while epoch < p.num_epochs:
        if rank == 0:
            logging.info(">>>>> start training in epoch %d" % epoch)
        if epoch >= p.sorta_epoch:
            trainset_builder.batch_wise_shuffle(p.batch_size)
        solver.train(trainset_builder.as_dataset(p.batch_size, p.num_data_threads))

        if p.devset_config is not None:
            if rank == 0:
                logging.info(">>>>> start evaluate in epoch %d" % epoch)
            devset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.devset_config)
            devset = devset_builder.as_dataset(p.batch_size, p.num_data_threads)
            loss_g, metrics_g = solver.evaluate(devset, epoch)

        if rank == 0:
            checkpointer(loss_g, metrics_g)

        epoch = epoch + 1


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)

    json_file = sys.argv[1]
    config = None
    with open(json_file) as f:
        config = json.load(f)
    p = parse_config(config)
    GanSolver.initialize_devices(p.solver_gpu)
    train(json_file, GanSolver, 1, 0)
