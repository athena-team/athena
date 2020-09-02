# Speech Features - User Manual

## Introduction

Transform is a preprocess data toolkit.

## Usage

Transform supports speech feature extract.

### 1. Import module

```python
from athena.transform import AudioFeaturizer
```

### 2. Init a feature extract object and extract features

#### ReadWav

By using `ReadWav`, we can get audio data and sample rate from a wavfile.

```python
config = {'type': 'ReadWav'}
audio_data, sample_rate = AudioFeaturizer(config)

'''
The audio data range from [-32768, 32767].

The shape of the output: [T, 1, 1].
'''
```

#### Spectrum

Using `Spectrum` , we can get a complex matrix with dimensions `(num_frequency, num_frame)`, which means that the signal is composed of num_frequency sine waves with different 
phases and magnitudes. For each T-F bin, the absolute value of the FFT is the magnitude, and the phase is the initial phase of the sine wave. The corresponding spectrum is called 
the amplitude spectrum and the phase spectrum, as shown in the figure below.

```python
config = {'type': 'Spectrum',
          'output_type': 2,
          'window_length': 0.025, 
          'frame_length': 0.010, 
          'preEph_coeff': 0.97,
          'window_type': 'hamming'}
spectrum_data = AudioFeaturizer(config)

'''
You can get power spectrum ('output_type': 1), log-power spectrum ('output_type': 2) or 
magnitude spectrum ('output_type': 3).

The shape of the output: [T, dim, 1]
'''
```

#### Fbank

Using `Fbank` , we can get a float matrix with dimensions `(num_frame, num_fbank, num_channel)`, which means the energy 
in the Mel frequency band to log. The Mel-frequency bands aim to mimic the non-linear human ear perception of sound, by being more 
discriminative at lower frequencies and less discriminative at higher frequencies. Each filter in the filter bank is 
triangular having a response of 1 at the center frequency and decrease linearly towards 0 till it reaches the center 
frequencies of the two adjacent filters where the response is 0. After applying the filter bank to the power spectrum 
(periodogram) of the signal, we obtain the fbank feature.


```python
config = {'type': 'Fbank',
          'delta_delta': False,
          'lower_frequency_limit': 100,
          'upper_frequency_limit': 0,
          'filterbank_channel_count': 80}
fbank_data = AudioFeaturizer(config)

'''
You can get delta fbank by setting 'delta_delta' to 'Ture'. The channel of fbank is equal to
'order' + 1 ('delta_delta': True) or 1 ('delta_delta': False).

The shape of the output: [T, dim, num_channel].
'''
```

#### Pitch

Using `Pitch` , we can get a float matrix with dimensions `(num_frame, 2)`, which is consisting of `(NCCF, pitch in Hz)` of 
each frame. `NCCF` is the Normalized Cross Correlation Function, which is related to the probability of voicing and which helps
in ASR. `Pitch` is the fundamental frequency (F0) of voice, which can get large performance improvements on tonal languages for ASR.

```python
config = {'type': 'Pitch',
          'window_length': 0.025, 
          'soft_min_f0': 10.0,
          'lowpass-cutoff': 1000,
          'max-f0': 400}
pitch_data = AudioFeaturizer(config)

'''
You can get more information about pitch by "Ghahremani P, BabaAli B, Povey D, et al. A pitch extraction algorithm tuned 
for automatic speech recognition[C]//2014 IEEE international conference on acoustics, speech and signal processing (ICASSP).
 IEEE, 2014: 2494-2498."

The shape of the output: [T, 2].
'''

```

#### Mfcc

Using `Mfcc` , we can get a float matrix with dimensions `(num_frame, num_coefficient)`.
Mel-Frequency Cepstral Coefficients (MFCC) is a cepstrum extracted in the frequency domain of the Mel scale. 
It is a feature widely used in automatic speech and speaker recognition. It turns out that filter bank coefficients computed
in the fbank are highly correlated, which could be problematic in some machine learning algorithms. Therefore, we can 
apply Discrete Cosine Transform (DCT) to decorrelate the filter bank coefficients and yield a compressed representation 
of the filter banks. 

```python
config = {'type': 'Mfcc',
          'coefficient_count': 13, 
          'cepstral_lifter': 22, 
          'use_energy': True }
mfcc_data = AudioFeaturizer(config)

'''
Typically, for ASR, the resulting cepstral coefficients 2-13 are 
retained and the rest are discarded.

The shape of the output: [T, dim].
'''
```

#### MelSpectrum

Using `MelSpectrum` , we can get a float matrix with dimensions `(num_frame, num_filterbank)`. `Melspectrum` is
applying triangular filters on a Mel-scale to the magnitude spectrum to extract frequency bands, which based on MelSpectrum of Librosa.
However, `Fbank` is mainly based on Kaldi. `MelSpectrum` is a feature widely used in TTS.

```python
config = {'type': 'MelSpectrum',
          'output_type': 1,
          'lower_frequency_limit': 60,
          'filterbank_channel_count': 40}
mel_spectrum_data = AudioFeaturizer(config)

'''
The mel_spectrum features can be inversely transformed to waveform by vocoder.

The shape of the output: [T, dim].
'''
```

#### CMVN

Using  `CMVN`, we can do Cepstral Mean and Variance Normalization on features.

```python
config = {'type': 'CMVN',
          'global_mean': 0.0,
          'global_variance': 1.0}
cmvn_features = AudioFeaturizer(config)
```

### 3. Get feature dim and the number of channels

```python
dim = feature_ext.dim
# dim : A int scalar, it is feature dim.
num_channels = feature_ext.num_channels
# num_channels: A int scalar, it is the number of channels of the output feature
# the shape of the output features: [None, dim, num_channels]
```

### 4. Example

#### 4.1 Extract speech feature filterbank from audio file:

```python
import tensorflow as tf
from transform.audio_featurizer import AudioFeaturizer

audio_file = 'englist.wav'
config = {'type': 'Fbank',
          'sample_rate': 8000,
          'delta_delta': True}
feature_ext = AudioFeaturizer(config)
dim = feature_ext.dim
print('Dim size is ', dim)
num_channels = feature_ext.num_channels
print('num_channels is ', num_channels)

feat = feature_ext(audio_file)
with tf.Session() as sess:
  fbank = sess.run(feat)
  print('Fbank shape is ', fbank.shape)
```

Result:

```
Dim is  40
Fbank shape is  (346, 40, 3) # [T, D, C]
```

#### 4.2 CMVN usage:

```python
import tensorflow as tf
from transform.audio_featurizer import AudioFeaturizer

dim = 23
config = {'type': 'CMVN',
          'global_mean': np.zeros(dim).tolist(),
          'global_variance': np.ones(dim).tolist(),
          'local_cmvn': True}
cmvn = AudioFeaturizer(config)
audio_feature = tf.compat.v1.random_uniform(shape=[10, dim], dtype=tf.float32, maxval=1.0)
print('cmvn : ', cmvn(audio_feature))
```
