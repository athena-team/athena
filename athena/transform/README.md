# speech featurizer

## 1. Read wave

Read audio sample from wav file, return sample data and sample rate.

### 1.1 read_wav.py

#### 1.1.1 Usage
```python
from athena.transform.feats.readwav import Readwav
conf = {'audio_channels': 1}
speed = 0.9
readwav = Readwav.params(conf).instantiate()
audio_data, sample_rate = readwav(filepath, speed)
```
    
#### 1.1.2 Configures Setting[Options]

```python
"audio_channels"     : Number of sample channels wanted. (int, default = 1)
``` 

### 1.2 write_wav.py
#### 1.2.1 Usage 

```python
from athena.transform.feats.write_wav import WriteWav
with tf.cached_session() as sess:
    write_wav = WriteWav.params().instantiate()
    writewav_op = write_wav(new_path, input_data, sample_rate)
    sess.run(writewav_op)
```
        
#### 1.2.2 Configures Setting[Options]

```python
"sample_rate"        : Sample frequency of waveform data. (int, default = 16000)
```

## 2. Spectrum

Audio signal gets sliced into (overlapping) frames and a window function is applied to each frame.
Afterwards, we do a Fourier transform on each frame (or more specifically a Short-Time Fourier Transform) and calculate the power spectrum.

### 2.1 spectrum.py

#### 2.1.1 Usage

```python
from athena.transform.feats.spectrum import Spectrum
conf = {'window_length':0.025, 'output_type':1}
spectrum = Spectrum.params(conf).instantiate()
spectrum_feats = spectrum(input_data, sample_rate)
```

#### 2.1.2 Configures Setting [Options]

```python
"window_length"       : "Window length in seconds. (float, default = 0.025)"
"frame_length"        : "Hop length in seconds. (float, default = 0.010)"
"snip_edges"          : "If 1, the last frame (shorter than window_length) will be cutoff. If 2, 1 // 2 frame_length data will be padded to data. (int, default = 1)"
"raw_energy"          : "If 1, compute frame energy before preemphasis and windowing. If 2, compute frame energy after preemphasis and windowing. (int, default = 1)"
"preEph_coeff"        : "Coefficient for use in frame-signal preemphasis. (float, default = 0.97)"
"window_type"         : "Type of window ('hamm'|'hann'|'povey'|'rect'|'blac'|'tria'). (string, default='povey')"
"remove_dc_offset"    : "Subtract mean from waveform on each frame (bool, default = true)"
"is_fbank"            : "If true, compute power spetrum without frame energy. If false, using the frame energy instead of the square of the constant component of the signal. (bool, default = false)"
"output_type"         : "If 1, return power spectrum. If 2, return log-power spectrum. (int, default = 2)"
"dither"              : "Dithering constant (0.0 means no dither) (float, default = 1) [add robust to training]"
```

## 3. FilterBank

Computing filter banks is applying triangular filters on a Mel-scale to the power spectrum to extract frequency bands.

### 3.1 fbank.py

#### 3.1.1 Usage

```python
from athena.transform.feats.fbank import Fbank
conf = {'delta_delta':False, 'window_length':0.025, 'output_type':1}
fbank = Fbank.params(conf).instantiate()
fbank_feats = fbank(input_data, sample_rate)
```

#### 3.1.2 Configures Setting [Options]

```python
"window_length"         : "Window length in seconds. (float, default = 0.025)"
"frame_length"          : "Hop length in seconds. (float, default = 0.010)"
"snip_edges"            : "If 1, the last frame (shorter than window_length) will be cutoff. If 2, 1 // 2 frame_length data will be padded to data. (int, default = 1)"
"raw_energy"            : "If 1, compute frame energy before preemphasis and windowing. If 2, compute frame energy after preemphasis and windowing. (int, default = 1)"
"preEph_coeff"          : "Coefficient for use in frame-signal preemphasis. (float, default = 0.97)"
"window_type"           : "Type of window ('hamm'|'hann'|'povey'|'rect'|'blac'|'tria'). (string, default='povey')"
"remove_dc_offset"      : "Subtract mean from waveform on each frame (bool, default = true)"
"is_fbank"              : "If true, compute power spetrum without frame energy. If false, using the frame energy instead of the square of the constant component of the signal. (bool, default = true)"
"output_type"           : "If 1, return power spectrum. If 2, return log-power spectrum. (int, default = 1)"
"upper_frequency_limit" : "High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (float, default = 0)"
"lower_frequency_limit" : "Low cutoff frequency for mel bins (float, default = 20)"
"filterbank_channel_count" : "Number of triangular mel-frequency bins (float, default = 23)"
"dither"                : "Dithering constant (0.0 means no dither) (float, default = 1) [add robust to training]"

#TODO
"use-energy"            : "Add an extra dimension with energy to the FBANK output. (bool, default = false)"
```

## 4. Pitch

This model extracts (pitch, NCCF) per frame. [Normalized Cross Correlation Function (NCCF)]

### 4.1 pitch.py

#### 4.1.1 Usage

```python
from athena.transform.feats.pitch import Pitch
conf = {'window_length':0.025, 'soft_min_f0':10.0}
pitch = Pitch.params(conf).instantiate()
pitch_test = pitch(input_data, sample_rate)
```

#### 4.1.2 Configures Setting[Options]

```python
"delta-pitch"             : "Smallest relative change in pitch that our algorithm measures (float, default = 0.005)"
"frame-length"            : "Frame length in seconds (float, default = 0.025)"
"frame-shift"             : "Frame shift in seconds (float, default = 0.010)"
"frames-per-chunk"        : "Only relevant for offline pitch extraction (e.g. compute-kaldi-pitch-feats), you can set it to a small nonzero value, such as 10, for better feature compatibility with online decoding (affects energy normalization in the algorithm) (int, default = 0)"
"lowpass-cutoff"          : "cutoff frequency for LowPass filter (Hz) (float, default = 1000)"
"lowpass-filter-width"    : "Integer that determines filter width of lowpass filter, more gives sharper filter (int, default = 1)"
"max-f0"                  : "max. F0 to search for (Hz) (float, default = 400)"
"max-frames-latency"      : "Maximum number of frames of latency that we allow pitch tracking to introduce into the feature processing (affects output only if frames-per-chunk > 0 and simulate-first-pass-online=true (int, default = 0)"
"min-f0"                  : "min. F0 to search for (Hz) (float, default = 50)"
"nccf-ballast"            : "Increasing this factor reduces NCCF for quiet frames (float, default = 7000)"
"nccf-ballast-online"     : "This is useful mainly for debug; it affects how the NCCF ballast is computed. (bool, default = false)"
"penalty-factor"          : "cost factor for FO change. (float, default = 0.1)"
"preemphasis-coefficient" : "Coefficient for use in signal preemphasis (deprecated) (float, default = 0)"
"recompute-frame"         : "Only relevant for online pitch extraction, or for compatibility with online pitch extraction. A non-critical parameter; the frame at which we recompute some of the forward pointers, after revising our estimate of the signal energy. Relevant if--frames-per-chunk > 0 (int, default = 500)"
"resample-frequency"      : "Frequency that we down-sample the signal to. Must be more than twice lowpass-cutoff (float, default = 4000)"
"simulate-first-pass-online" : "If true, compute-kaldi-pitch-feats will output features that correspond to what an online decoder would see in the first pass of decoding-- not the final version of the features, which is the default. Relevant if frames-per-chunk > 0 (bool, default = false)"
"snip-edges"              : "If this is set to false, the incomplete frames near the ending edge won't be snipped, so that the number of frames is the file size divided by the frame-shift. This makes different types of features give the same number of frames. (bool, default = true)"
"soft-min-f0"             : "Minimum f0, applied in soft way, must not exceed min-f0 (float, default = 10)"
"upsample-filter-width"   : "Integer that determines filter width when upsampling NCCF (int, default = 5)"
```

## 5. MFCC

This model extracts MFCC features per frame.

### 5.1 mfcc.py

#### 5.1.1 Usage

```python
from athena.transform.feats.mfcc import Mfcc
config = {'use_energy':True}
mfcc = Mfcc.params(conf).instantiate()
mfcc_test = mfcc(input_data, sample_rate)
```

#### 5.1.2 Configures Setting [Options]

```python
"window_length"              : "Window length in seconds. (float, default = 0.025)"
"frame_length"               : "Hop length in seconds. (float, default = 0.010)"
"snip_edges"                 : "If 1, the last frame (shorter than window_length) will be cutoff. If 2, 1 // 2 frame_length data will be padded to data. (int, default = 1)"
"-raw_energy"                : "If 1, compute frame energy before preemphasis and windowing. If 2, compute frame energy after preemphasis and windowing. (int, default = 1)"
"preEph_coeff"               : "Coefficient for use in frame-signal preemphasis. (float, default = 0.97)"
"window_type"                : "Type of window ('hamm'|'hann'|'povey'|'rect'|'blac'|'tria'). (string, default='povey')"
"remove_dc_offset"           : "Subtract mean from waveform on each frame (bool, default = true)"
"is_fbank"                   : "If true, compute power spetrum without frame energy. If false, using the frame energy instead of the square of the constant component of the signal. (bool, default = true)"
"output_type"                : "If 1, return power spectrum. If 2, return log-power spectrum. If 3, return magnitude spectrum. (int, default = 1)"
"upper_frequency_limit"      : "High cutoff frequency for mel bins (if < 0, offset from Nyquist) (float, default = 0)"
"lower_frequency_limit"      : "Low cutoff frequency for mel bins (float, default = 20)"
"filterbank_channel_count"   : "Number of triangular mel-frequency bins (float, default = 23)"
"coefficient_count"          : "Number of cepstra in MFCC computation.(int, default = 13)"
"cepstral_lifter"            : "Constant that controls scaling of MFCCs.(float, default = 22)"
"use_energy"                 : "Use energy (not C0) in MFCC computation. (bool, default = True)”
```
## 6. Fbank_Pitch

This model extracts Fbank && Pitch features per frame.

### 6.1 fbank_pitch.py

### 6.1.1 Usage

```python
from athena.transform.feats.fbank_pitch import FbankPitch
conf = {'window_length': 0.025, 'output_type': 1, 'frame_length': 0.010, 'dither': 0.0}
fbank_pitch = FbankPitch.params(conf).instantiate()
mfcc_test = fbank_pitch(input_data, sample_rate)
```
### 6.1.2 Configures Setting [Options]

```python
"window_length"		          : Window length in seconds. (float, default = 0.025)
"frame_length"			      : Hop length in seconds. (float, default = 0.010)
"snip_edges"			      : If 1, the last frame (shorter than window_length) will be cutoff. If 2, 1 // 2 frame_length data will be padded to data. (int, default = 1)
"raw_energy"				  : If 1, compute frame energy before preemphasis and windowing. If 2,  compute frame energy after preemphasis and windowing. (int, default = 1)
"preEph_coeff"			      : Coefficient for use in frame-signal preemphasis. (float, default = 0.97)
"window_type"				  : Type of window ("hamm"|"hann"|"povey"|"rect"|"blac"|"tria"). (string, default = "povey")
"remove_dc_offset"	          : Subtract mean from waveform on each frame (bool, default = true)
"is_fbank"				      : If true, compute power spetrum without frame energy. If false, using the frame energy instead of the square of the constant component of the signal. (bool, default = true)
"output_type"				  : If 1, return power spectrum. If 2, return log-power spectrum. (int, default = 1)
"upper_frequency_limit"	      : High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (float, default = 0)
"lower_frequency_limit"	      : Low cutoff frequency for mel bins (float, default = 20)
"filterbank_channel_count"    : Number of triangular mel-frequency bins (float, default = 23)
"dither"			    	  : Dithering constant (0.0 means no dither) (float, default = 1) [add robust to training]
"delta-pitch"                 : Smallest relative change in pitch that our algorithm measures (float, default = 0.005)
"frames-per-chunk"            : Only relevant for offline pitch extraction (e.g. compute-kaldi-pitch-feats), you can set it to a small nonzero value, such as 10, for better feature compatibility with online decoding (affects energy normalization in the algorithm) (int, default = 0)
"lowpass-cutoff"              : cutoff frequency for LowPass filter (Hz)  (float, default = 1000)
"lowpass-filter-width"        : Integer that determines filter width of lowpass filter, more gives sharper filter (int, default = 1)
"max-f0"                      : max. F0 to search for (Hz) (float, default = 400)
"max-frames-latency"          : Maximum number of frames of latency that we allow pitch tracking to introduce into the feature processing (affects output only if --frames-per-chunk > 0 and --simulate-first-pass-online=true (int, default = 0)
"min-f0"                      : min. F0 to search for (Hz) (float, default = 50)
"nccf-ballast"                : Increasing this factor reduces NCCF for quiet frames (float, default = 7000)
"nccf-ballast-online"         : This is useful mainly for debug; it affects how the NCCF ballast is computed. (bool, default = false)
"penalty-factor"              : cost factor for FO change. (float, default = 0.1)
"preemphasis-coefficient"     : Coefficient for use in signal preemphasis (deprecated) (float, default = 0)
"recompute-frame"             : Only relevant for online pitch extraction, or for compatibility with online pitch extraction.  A non-critical parameter; the frame at which we recompute some of the forward pointers, after revising our estimate of the signal energy.  Relevant if--frames-per-chunk > 0 (int, default = 500)
"resample-frequency"          : Frequency that we down-sample the signal to.  Must be more than twice lowpass-cutoff (float, default = 4000)
"simulate-first-pass-online"  : If true, compute-kaldi-pitch-feats will output features that correspond to what an online decoder would see in the first pass of decoding-- not the final version of the features, which is the default.  Relevant if --frames-per-chunk > 0 (bool, default = false)
"soft-min-f0"                 : Minimum f0, applied in soft way, must not exceed min-f0 (float, default = 10)
"upsample-filter-width"       : Integer that determines filter width when upsampling NCCF (int, default = 5)
```

## 7. MelSpectrum

This model extracts log melspectrum per frame.

### 7.1 melspectrum.py

#### 7.1.1 Usage

```python
from athena.transform.feats.mel_spectrum import MelSpectrum
conf = {'delta_delta':False, 'window_length':0.025}
melspectrum = MelSpectrum.params(conf).instantiate()
fbank_feats = melspectrum(input_data, sample_rate)
```

#### 7.1.2 Configures Setting [Options]

```python
"window_length"         : "Window length in seconds. (float, default = 0.025)"
"frame_length"          : "Hop length in seconds. (float, default = 0.010)"
"snip_edges"            : "If 1, the last frame (shorter than window_length) will be cutoff. If 2, 1 // 2 frame_length data will be padded to data. (int, default = 1)"
"raw_energy"            : "If 1, compute frame energy before preemphasis and windowing. If 2, compute frame energy after preemphasis and windowing. (int, default = 1)"
"preEph_coeff"          : "Coefficient for use in frame-signal preemphasis. (float, default = 0.0)"
"window_type"           : "Type of window ('hamm'|'hann'|'povey'|'rect'|'blac'|'tria'). (string, default='hann')"
"remove_dc_offset"      : "Subtract mean from waveform on each frame (bool, default = false)"
"is_fbank"              : "If true, compute power spetrum without frame energy. If false, using the frame energy instead of the square of the constant component of the signal. (bool, default = true)"
"output_type"           : "If 1, return power spectrum. If 2, return log-power spectrum. If 3, return magnitude spectrum. (int, default = 3)"
"upper_frequency_limit" : "High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (float, default = 0)"
"lower_frequency_limit" : "Low cutoff frequency for mel bins (float, default = 20)"
"filterbank_channel_count" : "Number of triangular mel-frequency bins (float, default = 23)"
"dither"                : "Dithering constant (0.0 means no dither) (float, default = 0) [add robust to training]"

```