# Speech Features - User Manual

## Introduction

Transform is a preprocess data toolkit.

## Usage

Transform support speech feature extract:

### 1. Import module

```python
from athena.transform import AudioFeaturizer
```

### 2. Init a feature extract object

#### Read wav

```python
conf = {'type':'ReadWav'}
feature_ext = AudioFeaturizer()

'''
Other default args:
'audio_channels': Number of sample channels wanted. (int, default = 1)

The shape of the output:
[T, 1, 1]
'''
```

#### Spectrum

```python
conf = {'type':'Spectrum'}
feature_ext = AudioFeaturizer(conf)

'''
Other default args:
'window_length' : Window length in seconds. (float, default = 0.025)
'frame_length' : Hop length in seconds. (float, default = 0.010)
'raw_energy' : If 1, compute frame energy before preemphasis and windowing. If 2,  compute frame energy after preemphasis and windowing. (int, default = 1)
'preEph_coeff' : Coefficient for use in frame-signal preemphasis. (float, default = 0.97)
'window_type' : Type of window ("hamm"|"hann"|"povey"|"rect"|"blac"|"tria"). (string, default = "povey")
'remove_dc_offset' : Subtract mean from waveform on each frame (bool, default = true)
'is_fbank' : If true, compute power spetrum without frame energy. If false, using the frame energy instead of the square of the constant component of the signal. (bool, default = false)
'output_type' : If 1, return power spectrum. If 2, return log-power spectrum. (int, default = 2)
'dither' : Dithering constant (0.0 means no dither) (float, default = 1) [add robust to training]
'global_mean': 0.0
'global_variance': 0.0
'local_cmvn' : True

The shape of the output:
[T, dim, 1]
'''
```

#### FliterBank

```python
conf = {'type':'Fbank'}
feature_ext = AudioFeaturizer(conf)

'''
Other default args:
'window_length'	: Window length in seconds. (float, default = 0.025)
'frame_length'	: Hop length in seconds. (float, default = 0.010)
'snip_edges' : If 1, the last frame (shorter than window_length) will be cutoff. If 2, 1 // 2 frame_length data will be padded to data. (int, default = 1)
'raw_energy' : If 1, compute frame energy before preemphasis and windowing. If 2,  compute frame energy after preemphasis and windowing. (int, default = 1)
'preEph_coeff' : Coefficient for use in frame-signal preemphasis. (float, default = 0.97)
'window_type' : Type of window ("hamm"|"hann"|"povey"|"rect"|"blac"|"tria"). (string, default = "povey")
'remove_dc_offset' : Subtract mean from waveform on each frame (bool, default = true)
'is_fbank': If true, compute power spetrum without frame energy. If false, using the frame energy instead of the square of the constant component of the signal. (bool, default = true)
'output_type' : If 1, return power spectrum. If 2, return log-power spectrum. (int, default = 1)
'upper_frequency_limit'	: High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (float, default = 0)
'lower_frequency_limit'	: Low cutoff frequency for mel bins (float, default = 20)
'filterbank_channel_count' : Number of triangular mel-frequency bins (float, default = 23)
'dither'	: Dithering constant (0.0 means no dither) (float, default = 1) [add robust to training]
'delta_delta' : False
'window' : 2
'order' : 2
'global_mean': 0.0
'global_variance': 0.0
'local_cmvn' : True

Returns:
  A tensor of shape [T, dim, num_channel].
  dim = 40
  num_channel = 1 if 'delta_delta' == False else 1 + 'order'
'''
```

#### Pitch
```python
conf = {'type':'Pitch'}
feature_ext = AudioFeaturizer(conf)
'''
Other default args:
'delta-pitch' : Smallest relative change in pitch that our algorithm measures (float, default = 0.005)
'frame-length' : Frame length in seconds (float, default = 0.025)
'frame-shift' : Frame shift in seconds (float, default = 0.010)
'frames-per-chunk' : Only relevant for offline pitch extraction (e.g. compute-kaldi-pitch-feats), you can set it to a small nonzero value, such as 10, for better feature compatibility with online decoding (affects energy normalization in the algorithm) (int, default = 0)
'lowpass-cutoff' : cutoff frequency for LowPass filter (Hz)  (float, default = 1000)
'lowpass-filter-width' : Integer that determines filter width of lowpass filter, more gives sharper filter (int, default = 1)
'max-f0' : max. F0 to search for (Hz) (float, default = 400)
'max-frames-latency': Maximum number of frames of latency that we allow pitch tracking to introduce into the feature processing (affects output only if --frames-per-chunk > 0 and --simulate-first-pass-online=true (int, default = 0)
'min-f0' : min. F0 to search for (Hz) (float, default = 50)
'nccf-ballast' : Increasing this factor reduces NCCF for quiet frames (float, default = 7000)
'nccf-ballast-online'  : This is useful mainly for debug; it affects how the NCCF ballast is computed. (bool, default = false)
'penalty-factor' : cost factor for FO change. (float, default = 0.1)
'preemphasis-coefficient' : Coefficient for use in signal preemphasis (deprecated) (float, default = 0)
'ecompute-frame' : Only relevant for online pitch extraction, or for compatibility with online pitch extraction.  A non-critical parameter; the frame at which we recompute some of the forward pointers, after revising our estimate of the signal energy.  Relevant if--frames-per-chunk > 0 (int, default = 500)
'resample-frequency' : Frequency that we down-sample the signal to.  Must be more than twice lowpass-cutoff (float, default = 4000)
'simulate-first-pass-online' : If true, compute-kaldi-pitch-feats will output features that correspond to what an online decoder would see in the first pass of decoding-- not the final version of the features, which is the default.  Relevant if --frames-per-chunk > 0 (bool, default = false)
'snip-edges': If this is set to false, the incomplete frames near the ending edge won't be snipped, so that the number of frames is the file size divided by the frame-shift. This makes different types of features give the same number of frames. (bool, default = true)
'soft-min-f0' : Minimum f0, applied in soft way, must not exceed min-f0 (float, default = 10)
'upsample-filter-width': Integer that determines filter width when upsampling NCCF (int, default = 5)
'add-delta-pitch' : If true, time derivative of log-pitch is added to output features. (bool, default = true)
'add-pov-feature' : If true, the warped NCCF is added to output features. (bool, default = true)
'add-raw-log-pitch' : If true, log(pitch) is added to output features. (bool, default = false)
'delay' : Number of frames by which the pitch information is delayed. (int, default = 0)
'elta-pitch-noise-stddev' : Standard deviation for noise we add to the delta log-pitch (before scaling); should be about the same as delta-pitch option to pitch creation.  The purpose is
                                    to get rid of peaks in the delta-pitch caused by discretization of pitch values. (float, default = 0.005)
'delta-pitch-scale' : Term to scale the final delta log-pitch feature. (float, default = 10)
'delta-window' : Number of frames on each side of central frame, to use for delta window. (int, default = 2)
'normalization-left-context' : Left-context (in frames) for moving window normalization. (int, default = 75)
'normalization-right-context' : Right-context (in frames) for moving window normalization. (int, default = 75)
'pitch-scale' : Scaling factor for the final normalized log-pitch value. (float, default = 2)
'pov-offset' : This can be used to add an offset to the POV feature. Intended for use in online decoding as a substitute for  CMN. (float, default = 0)
'pov-scale' : Scaling factor for final POV (probability of voicing) feature. (float, default = 2)
'''
```

#### MFCC
```python
conf = {'type':'Mfcc'}
feature_ext = AudioFeaturizer(conf)
'''
'window_lengtt': Window length in seconds. (float, default = 0.025)
'frame_length' : Hop length in seconds. (float, default = 0.010)
'snip_edges' : If 1, the last frame (shorter than window_length) will be cutoff. If 2, 1 // 2 frame_length data will be padded to data. (int, default = 1)
'raw_energy' : If 1, compute frame energy before preemphasis and windowing. If 2,  compute frame energy after preemphasis and windowing. (int, default = 1)
'preEph_coeff' : Coefficient for use in frame-signal preemphasis. (float, default = 0.97)
'window_type' : Type of window ("hamm"|"hann"|"povey"|"rect"|"blac"|"tria"). (string, default = "povey")
'remove_dc_offset' : Subtract mean from waveform on each frame (bool, default = true)
'is_fbank' : If true, compute power spetrum without frame energy. If false, using the frame energy instead of the square of the constant component of the signal. (bool, default = true)
'output_type' : If 1, return power spectrum. If 2, return log-power spectrum. (int, default = 1)
'upper_frequency_limit'	: High cutoff frequency for mel bins (if < 0, offset from Nyquist) (float, default = 0)
'lower_frequency_limit' : Low cutoff frequency for mel bins (float, default = 20)
'filterbank_channel_count'	: Number of triangular mel-frequency bins (float, default = 23)
'coefficient_count'  : Number of cepstra in MFCC computation.(int, default = 13)
'cepstral_lifter' : Constant that controls scaling of MFCCs.(float, default = 22)
'use_energy' :Use energy (not C0) in MFCC computation. (bool, default = True)
'''
```

#### MelSpectrum
```python
conf = {'type':'MelSpectrum'}
feature_ext = AudioFeaturizer(conf)
```

#### CMVN

```python
conf = {'type':'CMVN'}
cmvn = AudioFeaturizer(conf)

'''
Other configuration

'global_mean': global cmvn
'global_variance': global variance
'local_cmvn' : default true

'''
```

### 3. Feature extract

```python
feat =  feature_ext(audio)
'''
audio : Audio file or audio data.
feat : A tensor containing speech feature.
'''
```

### 4. Get feature dim and the number of channels

```python
dim = feature_ext.dim
# dim : A int scalar, it is feature dim.
num_channels = feature_ext.num_channels
# num_channels: A int scalar, it is the number of channels of the output feature
# the shape of the output features: [None, dim, num_channels]
```

### 5. Example

#### 5.1 Extract speech feature filterbank from audio file:

```python
import tensorflow as tf
from transform.audio_featurizer import AudioFeaturizer

audio_file = 'englist.wav'
conf = {'type':'Fbank'
        'sample_rate':8000,
        'delta_delta': True
       }
feature_ext = AudioFeaturizer(conf)
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

#### 5.2 CMVN usage:

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
