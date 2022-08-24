# coding=utf-8
# Copyright (C) ATHENA AUTHORS; Jianwei Sun; Cheng Wen
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
# pylint: disable=invalid-name, no-member, redefined-outer-name
r""" entry point for inference of all kinds of models """
import os
import tensorflow as tf
from athena.transform import AudioFeaturizer
from athena.data.feature_normalizer import FeatureNormalizer
from athena.data.text_featurizer import Vocabulary
from athena.data import SpeechSynthesisTestDatasetBuilder
import argparse
from absl import logging, flags
import soundfile as sf
import sys

class TTSTestSolver:

    default_config = {

        "text_config": {"type":"eng_vocab", "model":"examples/tts/data_baker/data/vocab"},
        "num_cmvn_workers": 1,
        "cmvn_file": None,
        "remove_unk": True,
        "input_length_range": [1, 10000],
        "output_length_range": [20, 50000],
        "data_csv": None,
    }
    def __init__(self, data_csv, saved_model_dir):
        aco_saved_model = os.path.join(saved_model_dir,"fastspeech2")
        if os.path.exists(aco_saved_model):
            self.aco_model = tf.keras.models.load_model(aco_saved_model)
        else:
            raise ValueError("Please specify fastspeech2 saved_model")

        vocoder_saved_model = os.path.join(saved_model_dir,"hifigan")
        if os.path.exists(vocoder_saved_model):
            self.vocoder = tf.keras.models.load_model(vocoder_saved_model)
        else:
            raise ValueError("Please specify hifigan saved_model")
        self.dataset = SpeechSynthesisTestDatasetBuilder(data_csv, self.default_config)

    def inference(self, out_dir):
        """ synthesize using vocoder on dataset """
        dataset = self.dataset.as_dataset(batch_size=1)

        synthesize_step = self.aco_model.deploy_function
        vocoder_step = self.vocoder.deploy_function
        for i, samples in enumerate(dataset):
            outputs = synthesize_step(samples['input'])
            features = outputs.numpy()[0]
            mel_lengths = features.shape[0]
            utt_id = samples['utt_id'].numpy()[0].decode()
            generated_audios = vocoder_step(outputs).numpy()[0]
            sf.write(
               os.path.join(out_dir, f"{utt_id}.wav"),
                generated_audios[: mel_lengths * 300],
                 24000,
                 "PCM_16",
            )


class ASRTestSolver:
    testset_config = {
        "data_csv": "examples/asr/aishell/data/test.csv",
        "audio_config": {"type": "Fbank", "filterbank_channel_count": 80},
        "cmvn_file": "examples/asr/aishell/data/cmvn",
        "global_cmvn": True,
        "text_config": {"type": "vocab", "model": "examples/asr/aishell/data/vocab"},
        "input_length_range": [0, 100000],
        "ignore_unk": True
    }
    def __init__(self,saved_model_dir):
        self.model = tf.keras.models.load_model(saved_model_dir)
        self.audio_featurizer = AudioFeaturizer(self.testset_config["audio_config"])
        self.feature_normalizer = FeatureNormalizer(self.testset_config["cmvn_file"])
        self.vocabulary = Vocabulary(self.testset_config["text_config"]["model"])

    def decode(self, wav_dir=None):
        if wav_dir != None:
            feat = self.audio_featurizer(wav_dir, speed=1.0)
            feat = self.feature_normalizer(feat, 'global')
            feat = tf.expand_dims(tf.convert_to_tensor(feat), 0)
            predictions = self.model.deploy_function(feat)
            predictions = self.vocabulary.decode_to_list(predictions.numpy().tolist(), [4231, 4231])
            predictions = ''.join(predictions[0])
        return predictions

    def inference(self, args=None):
        if args.inference_type == "asr":
            if args.wav_dir != None:
                predictions = self.decode(args.wav_dir)
                print(args.wav_dir + '\t' + predictions)
            elif args.wav_list != None:
                wav_list = open(args.wav_list, 'r')
                for wav_dir in wav_list:
                    wav_dir = wav_dir.strip('\n')
                    predictions = self.decode(wav_dir)
                    print(wav_dir + '\t' + predictions)
            else:
                logging.error("ERROR: can not get wav or wavlist")
                sys.exit()

if __name__=="__main__":
    logging.use_absl_handler()
    flags.FLAGS.mark_as_parsed()
    logging.set_verbosity(logging.INFO)
    parser = argparse.ArgumentParser(description='run a audio processing demo')
    parser.add_argument("--inference_type", type=str, required=True, choices=['asr', 'tts'], help="inference type")
    parser.add_argument("--saved_model_dir",type=str, required=True,help="saved model path")
    parser.add_argument("--wav_dir", type=str, default= None,help="audio path( audio type: pcm, wav)")
    parser.add_argument("--wav_list", type=str, default= None,help="batch decoding method: audio path list file")
    parser.add_argument("--text_csv", type=str, default= None,help="if inference_type is tts, text csv is required")
    parser.add_argument("--out_dir", type=str, default= "output",help="output dir")
    '''
    egs:
    wav_dir: xxx.wav
    wav list file:
    cat xxx1.wav xxx2.wav>wav.lst
    '''

    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.inference_type == "asr":
        test_solver = ASRTestSolver(args.saved_model_dir)
        test_solver.inference(args=args)
    
    if args.inference_type == "tts":
        if args.text_csv == None:
            raise ValueError("Please specify --text_csv")
        test_solver = TTSTestSolver(args.text_csv, args.saved_model_dir)
        test_solver.inference(args.out_dir)