# Copyright (C) ATHENA AUTHORS;Cheng Wen
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
import numpy as np
import pyworld as pw
import librosa
import soundfile as sf
import os
import pandas as pd
import tensorflow as tf
from athena.transform import AudioFeaturizer

class ExtractFeatures:
    """
        Refered https://github.com/TensorSpeech/TensorFlowTTS
    """
    def __init__(self, config):
        # hparams
        self.audio_config = config['audio_config']

        self.audio_featurizer = AudioFeaturizer(self.audio_config)

    def _is_outlier(self, x, p25, p75):
        """Check if value is an outlier."""
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        return x <= lower or x >= upper


    def _remove_outlier(self, x):  
        """Remove outlier from x."""
        p25 = np.percentile(x, 25)
        p75 = np.percentile(x, 75)

        indices_of_outliers = []
        for ind, value in enumerate(x):
            if self._is_outlier(value, p25, p75):
                indices_of_outliers.append(ind)

        x[indices_of_outliers] = 0.0

        # replace by mean f0.
        x[indices_of_outliers] = np.max(x)
        return x

    def extractor_features(self, wav_file, output_dir):
        """
        Extractor mel,f0 and energy and save in output_dir
        """
        config = self.audio_config
        utt_id = os.path.splitext(os.path.basename(wav_file))[0]
        # extract mel
        mel_tensor = self.audio_featurizer(wav_file)
        mel = mel_tensor.numpy()

        audio, sampling_rate = sf.read(wav_file)
        audio = audio.astype(np.float32)
        hop_length = int(config["frame_length"]*sampling_rate)
        win_length = int(config["window_length"]*sampling_rate)
        D = librosa.stft(
            audio,
            n_fft=2048,
            hop_length=hop_length,
            win_length=win_length,
            window=config["window_type"],
            pad_mode="reflect",
        )
        S, _ = librosa.magphase(D)  # (#bins, #frames)

        # check audio and feature length
        audio = np.pad(audio, (0, 2048), mode="edge")
        audio = audio[: len(mel) * hop_length]
        assert len(mel) * hop_length == len(audio)

        # extract raw pitch
        fmax = sampling_rate // 2 if config["upper_frequency_limit"] is None else config["upper_frequency_limit"]
        _f0, t = pw.dio(
            audio.astype(np.double),
            fs=sampling_rate,
            f0_ceil=fmax,
            frame_period=1000 * config["frame_length"],
        )
        f0 = pw.stonemask(audio.astype(np.double), _f0, t, sampling_rate)
        if len(f0) >= len(mel):
            f0 = f0[: len(mel)]
        else:
            f0 = np.pad(f0, (0, len(mel) - len(f0)))

        # extract energy
        energy = np.sqrt(np.sum(S ** 2, axis=0))
        if len(energy) >= len(mel):
            energy = energy[: len(mel)]
        else:
            energy = np.pad(energy, (0, len(energy) - len(energy)))

        assert len(mel) == len(f0) == len(energy)

        # remove outlier f0/energy
        f0 = self._remove_outlier(f0)
        energy = self._remove_outlier(energy)
        feat_path = os.path.join(output_dir, utt_id+ '.npz')
        np.savez(feat_path, mel= mel, energy= energy, f0= f0)
