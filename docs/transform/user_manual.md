# User Manual

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
'audio_channels':1

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
'sample_rate' : 16000
'window_length' : 0.025
'frame_length' : 0.010
'global_mean': 全局均值
'global_variance': 全局方差
'local_cmvn' : 默认是True, 做句子cmvn

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
'sample_rate' : 采样率 16000
'window_length' : 窗长 0.025秒
'frame_length' : 步长 0.010秒
'upper_frequency_limit' : 4000
'lower_frequency_limit': 20
'filterbank_channel_count'  : 40
'delta_delta' : 是否做差分 False
'window' : 差分窗长 2
'order' : 差分阶数 2
'global_mean': 全局均值
'global_variance': 全局方差
'local_cmvn' : 默认是True, 做句子cmvn

Returns:
  A tensor of shape [T, dim, num_channel].
  dim = 40
  num_channel = 1 if 'delta_delta' == False else 1 + 'order'
'''
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

'global_mean'和'global_variance'如果设置则会做全局cmvn，否则不做全局cmvn。
'local_cmvn' 设置False不做句子cmvn
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
