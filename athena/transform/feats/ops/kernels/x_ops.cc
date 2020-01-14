/* Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

// See tensorflow/core/ops/audio_ops.cc
namespace delta {
namespace {

using namespace tensorflow;  // NOLINT
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;
using tensorflow::shape_inference::UnchangedShape;


//SetShapeFn has issue [https://github.com/tensorflow/tensorflow/issues/29643]

Status SpectrumShapeFn(InferenceContext* c) {
  ShapeHandle input_data;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_data));

  int wav_len = c->Value(c->Dim(input_data, 0));
  int SampleRate = 16000;
  float win_len, frm_len;
  TF_RETURN_IF_ERROR(c->GetAttr("window_length", &win_len));
  TF_RETURN_IF_ERROR(c->GetAttr("frame_length", &frm_len));
  int i_WinLen = static_cast<int>(win_len * SampleRate);
  int i_FrmLen = static_cast<int>(frm_len * SampleRate);
  int i_NumFrm = (wav_len - i_WinLen) / i_FrmLen + 1;
  int i_FrqNum = static_cast<int>(pow(2.0f, ceil(log2(i_WinLen))) / 2 + 1);
  c->set_output(0, c->Matrix(i_NumFrm, i_FrqNum));

  return Status::OK();
}

Status PitchShapeFn(InferenceContext* c) {
  ShapeHandle input_data;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_data));

  int wav_len = c->Value(c->Dim(input_data, 0));
  int SampleRate = 16000;
  float win_len, frm_len;
  TF_RETURN_IF_ERROR(c->GetAttr("window_length", &win_len));
  TF_RETURN_IF_ERROR(c->GetAttr("frame_length", &frm_len));
  int i_WinLen = static_cast<int>(win_len * SampleRate);
  int i_FrmLen = static_cast<int>(frm_len * SampleRate);
  int i_NumFrm = (wav_len - i_WinLen) / i_FrmLen + 1;
  int i_FrqNum = static_cast<int>(pow(2.0f, ceil(log2(i_WinLen))) / 2 + 1);
  c->set_output(0, c->Matrix(i_NumFrm, 2));

  return Status::OK();
}

Status FbankShapeFn(InferenceContext* c) {
  ShapeHandle spectrogram;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &spectrogram));
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

  int32 filterbank_channel_count;
  TF_RETURN_IF_ERROR(
      c->GetAttr("filterbank_channel_count", &filterbank_channel_count));

  DimensionHandle audio_channels = c->Dim(spectrogram, 0);
  DimensionHandle spectrogram_length = c->Dim(spectrogram, 1);

  DimensionHandle output_channels = c->MakeDim(filterbank_channel_count);

  c->set_output(
      0, c->MakeShape({audio_channels, spectrogram_length, output_channels}));
  return Status::OK();
}



}  // namespace

// TF:
// wav to spectrogram:
// https://github.com/tensorflow/tensorflow/examples/wav_to_spectrogram/wav_to_spectrogram.h
// audio ops: https://github.com/tensorflow/tensorflow/core/ops/audio_ops.cc
// DecodeWav:
// tensorflow/tensorflow/core/api_def/base_api/api_def_DecodeWav.pbtxt
// EncodeWav:
// tensorflow/tensorflow/core/api_def/base_api/api_def_EncodeWav.pbtxt
// AudioSpectrogram:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/spectrogram_op.cc
//     tensorflow/tensorflow/core/api_def/base_api/api_def_AudioSpectrogram.pbtxt
// Mfcc:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/mfcc_op.cc
//     tensorflow/tensorflow/core/api_def/base_api/api_def_Mfcc.pbtxt

// TFLite
// mfcc:
// https://github.com/tensorflow/tensorflow/tensorflow/lite/kernels/internal/mfcc_mel_filterbank.h
// spectorgram: tensorflow/tensorflow/lite/kernels/internal/spectrogram.h

REGISTER_OP("Spectrum")
    .Input("input_data: float")
    .Input("sample_rate: float")
    .Attr("window_length: float = 0.025")
    .Attr("frame_length: float = 0.010")
    .Attr("window_type: string")
    .Attr("output_type: int = 2")
    .Attr("snip_edges: bool = true")
    .Attr("raw_energy: int = 1")
    .Attr("preEph_coeff: float = 0.97")
    .Attr("remove_dc_offset: bool = true")
    .Attr("is_fbank: bool = true")
    .Attr("dither: float = 1.0")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        return Status::OK();
    })
//    .SetShapeFn(SpectrumShapeFn)
    .Doc(R"doc(
    Create spectrum feature files.
    input_data: float, input wave, a tensor of shape [1, data_length].
    sample_rate: float, NB 8000, WB 16000 etc.
    window_length: float, window length in second.
    frame_length: float, frame length in second.
    output_type: int, 1: PSD, 2: log(PSD).
    raw_energy: int, 1: raw energy, 2: wined_energy.
    output: float, PSD/logPSD features, [num_Frame, num_Subband].
    )doc");

REGISTER_OP("FramePow")
    .Input("input_data: float")
    .Input("sample_rate: float")
    .Attr("snip_edges: bool = true")
    .Attr("remove_dc_offset: bool = true")
    .Attr("window_length: float = 0.025")
    .Attr("frame_length: float = 0.010")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        return Status::OK();
    })
    .Doc(R"doc(
    Create frame energy feature files.
    input_data: float, input wave, a tensor of shape [1, data_length].
    sample_rate: float, NB 8000, WB 16000 etc.
    window_length: float, window length in second.
    frame_length: float, frame length in second.
    output: float, frame energy features, [num_Frame].
    )doc");

REGISTER_OP("Pitch")
    .Input("input_data: float")
    .Input("sample_rate: int32")
    .Attr("window_length: float = 0.025")
    .Attr("frame_length: float = 0.010")
    .Attr("snip_edges: bool = true")
    .Attr("preemph_coeff: float = 0.0")
    .Attr("min_f0: float = 50")
    .Attr("max_f0: float = 400")
    .Attr("soft_min_f0: float = 10.0")
    .Attr("penalty_factor: float = 0.1")
    .Attr("lowpass_cutoff: float = 1000")
    .Attr("resample_freq: float = 4000")
    .Attr("delta_pitch: float = 0.005")
    .Attr("nccf_ballast: float = 7000")
    .Attr("lowpass_filter_width: int = 1")
    .Attr("upsample_filter_width: int = 5")
    .Attr("max_frames_latency: int = 0")
    .Attr("frames_per_chunk: int = 0")
    .Attr("simulate_first_pass_online: bool = false")
    .Attr("recompute_frame: int = 500")
    .Attr("nccf_ballast_online: bool = false")
    .Attr("pitch_scale: float = 2.0")
    .Attr("pov_scale: float = 2.0")
    .Attr("pov_offset: float = 0.0")
    .Attr("delta_pitch_scale: float = 10.0")
    .Attr("delta_pitch_noise_stddev: float = 0.005")
    .Attr("normalization_left_context: int = 75")
    .Attr("normalization_right_context: int = 75")
    .Attr("delta_window: int = 2")
    .Attr("delay: int = 0")
    .Attr("add_pov_feature: bool = true")
    .Attr("add_normalized_log_pitch: bool = true")
    .Attr("add_delta_pitch: bool = true")
    .Attr("add_raw_log_pitch: bool = true")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        return Status::OK();
    })
    .Doc(R"doc(
    Create pitch feature files.
    input_data: float, input wave, a tensor of shape [1, data_length].
    sample_rate: float, NB 8000, WB 16000 etc.
    window_length: float, window length in second.
    frame_length: float, frame length in second.
    output: float, pitch features, [num_Frame, 2].
    )doc");

REGISTER_OP("Speed")
    .Input("input_data: float")
    .Input("sample_rate: int32")
    .Input("resample_freq: int32")
    .Attr("lowpass_filter_width: int = 1")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        return Status::OK();
    })
    .Doc(R"doc(
    Create pitch feature files.
    input_data: float, input wave, a tensor of shape [1, data_length].
    sample_rate: float, NB 8000, WB 16000 etc.
    )doc");

REGISTER_OP("MfccDct")
    .Input("fbank: float")
    .Input("framepow: float")
    .Input("sample_rate: int32")
    .Attr("coefficient_count: int = 13")
    .Attr("cepstral_lifter: float = 22")
    .Attr("use_energy: bool = true")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        return Status::OK();
    })
    .Doc(R"doc(
Create MFCC feature files.
fbank: float, A tensor of shape  a tensor of shape [audio_channels, fbank_length, fbank_feat_dim].
sample_rate: int32, how many samples per second the source audio used. e.g. 16000, 8000.
coefficient_count: int, Number of cepstra in MFCC computation.
cepstral_lifter: float, Constant that controls scaling of MFCCs.
output: float, mfcc features, a tensor of shape [audio_channels, fbank_length, mfcc_feat_dim].
)doc");

REGISTER_OP("Fbank")
    .Input("spectrogram: float")
    .Input("sample_rate: int32")
    .Attr("upper_frequency_limit: float = 0")
    .Attr("lower_frequency_limit: float = 20")
    .Attr("filterbank_channel_count: int = 23")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        return Status::OK();
    })
//    .SetShapeFn(FbankShapeFn)
    .Doc(R"doc(
Create Mel-filter bank (FBANK) feature files.(
spectrogram
: float, A tensor of shape [audio_channels, spectrogram_length, spectrogram_feat_dim].
sample_rate: int32, how many samples per second the source audio used. e.g. 16000, 8000.
upper_frequency_limit: float, the highest frequency to use when calculating the ceptstrum.
lower_frequency_limit: float, the lowest frequency to use when calculating the ceptstrum.
filterbank_channel_count: int, resolution of the Mel bank used internally. 
output: float, fbank features, a tensor of shape [audio_channels, spectrogram_length, bank_feat_dim].
)doc");

// ref: https//github.com/kaldi-asr/kaldi/src/featbin/add-deltas.cc
REGISTER_OP("DeltaDelta")
    .Input("features: float")
    .Output("features_with_delta_delta: float")
    .Attr("order: int = 2")
    .Attr("window: int = 2")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        return Status::OK();
    })
//    .SetShapeFn([](InferenceContext* c) {
//      int order;
//      TF_RETURN_IF_ERROR(c->GetAttr("order", &order));
//
//      ShapeHandle feat;
//      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &feat));
//
//      DimensionHandle nframe = c->Dim(feat, 0);
//      DimensionHandle nfeat = c->Dim(feat, 1);
//
//      DimensionHandle out_feat;
//      TF_RETURN_IF_ERROR(c->Multiply(nfeat, order + 1, &out_feat));
//
//      // int input_len = c->Value(nfeat);
//      // int output_len = input_len * (order + 1);
//      // DimensionHandle out_feat = c->MakeDim(output_len);
//      c->set_output(0, c->Matrix(nframe, out_feat));
//      return Status::OK();
//    })
    .Doc(R"doc(
Add deltas (typically to raw mfcc or plp features).
features: A matrix of shape [nframe, feat_dim].
features_with_delta_delta: A matrix of shape [nframe, feat_dim * (order + 1)].
order: int, order fo delta computation.
window: a int, parameter controlling window for delta computation(actual window
    size for each delta order is 1 + 2*window).
)doc");

}  // namespace delta
