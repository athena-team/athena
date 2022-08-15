// Copyright (C) 2022 ATHENA DECODER AUTHORS; Rui Yan
// All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "e2e_context.h"

namespace athena {

AthenaStatus E2EContext::Init(const AMDataPtr &am, const LMDataPtr &lm,
                              const DecodeOptionsPtr &decode_options_ptr) {
  if (nullptr == am || nullptr == lm || nullptr == decode_options_ptr) {
    LOG(ERROR) << "decoder config is null.";
    return STATUS_ERROR;
  }
  lm_ = lm;
  am_ = am;
  feature_pipeline_ptr_.reset(new FeaturePipeline(decode_options_ptr->feature_pipeline_config));
  if (nullptr == feature_pipeline_ptr_) {
    LOG(ERROR) << "feature pipeline init failed.";
    return STATUS_ERROR;
  }
  feat_bins = decode_options_ptr->feature_pipeline_config.num_bins;
  return STATUS_OK;
}

void E2EContext::Reset() {
  feature_pipeline_ptr_->Reset();
  decode_result_.clear();
}

bool E2EContext::InvokeDecoder(const char *data, size_t len, bool last_frame) {
  if (len < 0 || nullptr == data) {
    LOG(ERROR) << "Recieved wrong data";
    return false;
  }
  decode_result_.clear();
  std::vector<int16_t> wav_data(reinterpret_cast<const int16_t *>(data),
                                reinterpret_cast<const int16_t *>(data) + len / sizeof(int16_t));
  feature_pipeline_ptr_->AcceptWaveform(std::vector<float>(wav_data.begin(), wav_data.end()));
  if (last_frame) {
    feature_pipeline_ptr_->SetInputFinished();
    int num_frames = feature_pipeline_ptr_->NumFramesReady();
    std::vector<float> feats;
    for (int i = 0; i < num_frames; ++i) {
      std::vector<float> feat;
      feature_pipeline_ptr_->GetFrame(i, &feat);
      feats.insert(feats.end(), feat.begin(), feat.end());
    }
    unsigned int frames_decode = feats.size() / feat_bins;
    std::tuple<std::string, cppflow::tensor>
        input_tensor("serving_default_x", cppflow::tensor(feats, {1, frames_decode, feat_bins, 1}));
    std::vector<cppflow::tensor> output_tensor =
        am_->GetAmNnet()->operator()(std::vector<std::tuple<std::string, cppflow::tensor> >(1, input_tensor),
                                     std::vector<std::string>(1, "StatefulPartitionedCall"));
    std::vector<int> predictions = output_tensor[0].get_data<int>();
    BuildDecodeResult(predictions);
  }
  return true;
}

void E2EContext::BuildDecodeResult(std::vector<int> &predictions) {
  decode_result_.clear();
  auto words = lm_->GetWords();
  DecodeResult result;
  for (const auto &id : predictions) {
    if (id != lm_->eos_ && id != lm_->sos_) {
      result.sentence += words[id];
    }
  }
  decode_result_.emplace_back(result);
}

} // namespace athena