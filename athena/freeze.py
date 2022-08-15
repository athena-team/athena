# coding=utf-8
# Copyright (C) ATHENA AUTHORS; Ruixiong Zhang; Yang Han; Jianwei Sun
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
# pylint: disable=invalid-name, no-member, redefined-outer-name
r""" entry point for inference of all kinds of models """
import sys
import json
import tensorflow as tf
import os
from absl import logging, flags
from athena.main import (
    parse_config,
    build_model_from_jsonfile,
    SUPPORTED_DATASET_BUILDER,
    SUPPORTED_OPTIMIZER,
    SUPPORTED_MODEL,
    parse_jsonfile
)
from athena import *



def freeze_e2e(jsonfile, freezed_model_dir):
    """ entry point for model inference, do some preparation work """

    p, checkpointer, _ = build_model_from_jsonfile(jsonfile)
    avg_num = 1 if 'model_avg_num' not in p.inference_config else p.inference_config['model_avg_num']
    if avg_num > 0:
        checkpointer.compute_nbest_avg(avg_num)
    assert p.testset_config is not None

    beam_size = p.inference_config['beam_size']
    feat_dims = p.testset_config['audio_config']['filterbank_channel_count']
    model = checkpointer.model
    if p.inference_config['decoder_type']=="beam_search":
        logging.info("beam search:")
        def freezed_inference(x):
            input_length = tf.shape(x)[1]
            samples = {"input":x,
                    "input_length":[input_length]}
            outputs = model.model.freeze_beam_search(samples, beam_size)
            return outputs
    elif p.inference_config['decoder_type']=="ctc_prefix_beam_search":
        logging.info("ctc prefix beam search:")
        def freezed_inference(x):
            input_length = tf.shape(x)[1]
            samples = {"input":x,
                    "input_length":[input_length]}
            outputs = model.model.freeze_ctc_prefix_beam_search(samples, model.ctc_final_layer, beam_size=beam_size)
            return outputs
    elif p.inference_config['decoder_type']=="freeze_ctc_probs":
        logging.info("freeze ctc probs:")
        def freezed_inference(x):
            input_length = tf.shape(x)[1]
            samples = {"input":x,
                    "input_length":[input_length]}
            outputs = model.model.freeze_ctc_probs(samples, model.ctc_final_layer, beam_size=beam_size)
            return outputs

    '''
    test block
    '''
    x = tf.ones([1,100,80,1])
    output = freezed_inference(x)
    '''
    freeze
    '''
    model.deploy_function = tf.function(freezed_inference, input_signature=[tf.TensorSpec(shape=[1, None, feat_dims, 1], dtype=tf.float32)])
    model.prepare_samples = checkpointer.model.prepare_samples
    tf.saved_model.save(obj=model, export_dir=freezed_model_dir)

def freeze_tts(json_file, freezed_model_dir):
    """ freeze TensorFlow model trained on Python to saved model,
    """
    p, checkpointer, _ = build_model_from_jsonfile(json_file)
    model = checkpointer.model
    def inference(x):
        samples = {"input": x}
        outputs = model.synthesize(samples)
        return outputs[0]

    model.deploy_function = tf.function(inference,
                                        input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
    tf.saved_model.save(obj=model, export_dir=freezed_model_dir)

def freeze2tflite(jsonfile, freezed_model_dir):

    p = parse_jsonfile(jsonfile)

    dataset_builder = SUPPORTED_DATASET_BUILDER[p.dataset_builder](p.testset_config)
    model = SUPPORTED_MODEL[p.model](
        data_descriptions=dataset_builder,
        config=p.model_config
    )

    ckpt_path = p.model_freeze["from_ckpt"]
    basename = os.path.basename(p.model_freeze["from_ckpt"])
    dirname = os.path.dirname(p.model_freeze["from_ckpt"])
    pb_dir = freezed_model_dir

    optimizer = SUPPORTED_OPTIMIZER[p.optimizer](p.optimizer_config)
    checkpointer = Checkpoint(
        checkpoint_directory=dirname,
        model=model,
        optimizer=optimizer
    )


    if not os.path.exists(ckpt_path+".index"):
        logging.info("{} do not exists".format(ckpt_path))
        return
    checkpointer.restore(ckpt_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model.tflite_model)

    if p.model_freeze["float16_quantization"]:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        lite_model = converter.convert()
        pb_path = os.path.join(pb_dir, basename+".16bit.tflite")
        open(pb_path, "wb").write(lite_model)
    elif p.model_freeze["dynamic_range_quantization"]:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        lite_model = converter.convert()
        pb_path = os.path.join(pb_dir, basename+".8bit.tflite")
        open(pb_path, "wb").write(lite_model)
    elif p.model_freeze["integer_quantization"]:
        ds = dataset_builder.as_dataset(1).take(100)

        def representative_dataset():
            '''
            the shape of training data is [batch, timestep, height, width, channel]
            when freeze model, we have to convert data to shape [batch, height, width, channel]
            besides, we have to keep the batch dims fixed
            that is why we use this format: data['input'][0,:10,:,:,:]
            '''
            for data in ds:
                yield [data['input'][0,:10,:,:,:]]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        lite_model = converter.convert()

        pb_path = os.path.join(pb_dir, basename+".fullint.tflite")
        open(pb_path, "wb").write(lite_model)
    elif p.model_freeze["integer_int16activations_quantization"]:
        ds = dataset_builder.as_dataset(1).take(100)

        def representative_dataset():
            '''
            the shape of training data is [batch, timestep, height, width, channel]
            when freeze model, we have to convert data to shape [batch, height, width, channel]
            besides, we have to keep the batch dims fixed
            that is why we use this format: data['input'][0,:10,:,:,:]
            '''
            for data in ds:
                yield [data['input'][0,:10,:,:,:]]

        converter.representative_dataset = representative_dataset
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]

        lite_model = converter.convert()

        pb_path = os.path.join(pb_dir, basename+".mix.tflite")
        open(pb_path, "wb").write(lite_model)
    else:
        lite_model = converter.convert()
        pb_path = os.path.join(pb_dir, basename+".tflite")
        open(pb_path, "wb").write(lite_model)

    logging.info("*********************************************")
    logging.info("Convert to tensorflow lite model successfully")
    logging.info("*********************************************")


def freeze_u2(jsonfile, freezed_model_dir):
    """ entry point for model inference, do some preparation work """

    p, checkpointer, dataset_builder = build_model_from_jsonfile(jsonfile)
    avg_num = 1 if 'model_avg_num' not in p.inference_config else p.inference_config['model_avg_num']
    if avg_num > 0:
        checkpointer.compute_nbest_avg(avg_num)
    assert p.testset_config is not None
    # model = checkpointer.model

    # beam_size = p.inference_config['beam_size']
    feat_dims = p.testset_config['audio_config']['filterbank_channel_count']
    num_encoder_layers = p.model_config['model_config']['num_encoder_layers']
    d_model = p.model_config['model_config']['d_model']

    model_encoder = checkpointer.model.encoder_forward_chunk_freeze.get_concrete_function(
        tf.TensorSpec(shape=[1, None, feat_dims, 1], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=[1, None, d_model], dtype=tf.float32),
        tf.TensorSpec(shape=[num_encoder_layers, 1, None, d_model], dtype=tf.float32),
        tf.TensorSpec(shape=[num_encoder_layers, 1, None, d_model], dtype=tf.float32))
    model_ctc = checkpointer.model.ctc_forward_chunk_freeze.get_concrete_function(
        tf.TensorSpec(shape=[1, None, d_model], dtype=tf.float32))
    model_encoder_ctc = checkpointer.model.encoder_ctc_forward_chunk_freeze.get_concrete_function(
        tf.TensorSpec(shape=[1, None, feat_dims, 1], dtype=tf.float32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=None, dtype=tf.int32),
        tf.TensorSpec(shape=[1, None, d_model], dtype=tf.float32),
        tf.TensorSpec(shape=[num_encoder_layers, 1, None, d_model], dtype=tf.float32),
        tf.TensorSpec(shape=[num_encoder_layers, 1, None, d_model], dtype=tf.float32))
    get_subsample_rate = checkpointer.model.get_subsample_rate.get_concrete_function()
    get_init = checkpointer.model.get_subsample_rate.get_concrete_function()

    signatures = {"serving_default": model_encoder_ctc,
                  "encoder": model_encoder,
                  "ctc": model_ctc,
                  "get_subsample_rate": get_subsample_rate,
                  "get_init": get_init}
    tf.saved_model.save(checkpointer.model, freezed_model_dir, signatures=signatures)

FreezeFunctions = {
    "speech_transformer": freeze_e2e,
    "speech_conformer": freeze_e2e,
    "speech_transformer_u2": freeze_u2,
    "speech_conformer_u2":freeze_u2,
    "fastspeech2": freeze_tts,
    "fastspeech": freeze_tts,
    "tacotron2": freeze_tts,
}

if __name__ == "__main__":
    logging.use_absl_handler()
    flags.FLAGS.mark_as_parsed()
    logging.set_verbosity(logging.INFO)
    
    if len(sys.argv) < 3:
        logging.warning('Usage: python {} config_json_file freezed_model_dir'.format(sys.argv[0]))
        sys.exit()
    tf.random.set_seed(1)

    jsonfile = sys.argv[1]
    with open(jsonfile) as file:
        config = json.load(file)
    p = parse_config(config)

    freezed_model_dir = sys.argv[2]
    if not os.path.exists(freezed_model_dir):
        os.makedirs(freezed_model_dir)
    rank_size = 1
    rank_index = 0
    try:
        import horovod.tensorflow as hvd
        hvd.init()
        rank_size = hvd.size()
        rank_index = hvd.rank()
    except ImportError:
        print("There is some problem with your horovod installation. \
               But it wouldn't affect single-gpu inference")
    if rank_size > 1:
        HorovodSolver.initialize_devices(p.solver_gpu)
    else:
        BaseSolver.initialize_devices(p.solver_gpu)

    if p.model == 'mtl_transformer_ctc':
        freeze = FreezeFunctions[p.model_config["model"]]
    else:
        freeze = FreezeFunctions[p.model]
    freeze(jsonfile, freezed_model_dir)
