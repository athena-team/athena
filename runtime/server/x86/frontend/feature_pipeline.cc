// Copyright (C) 2022 ATHENA DECODER AUTHORS; Yang Han
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

#include "frontend/feature_pipeline.h"

#include <algorithm>
#include <fstream>
#include <cmath>

namespace athena {

FeaturePipeline::FeaturePipeline(const FeaturePipelineConfig &config)
    : config_(config),
      feature_dim_(config.num_bins),
      features_(-1),//cache all the features
      fbank_(config.num_bins, config.sample_rate, config.frame_length,
             config.frame_shift),
      num_frames_(0),
      input_finished_(false) {
  if (!config_.cmvn_file.empty())
    LoadCmvn(config_.cmvn_file);
}

void FeaturePipeline::AcceptWaveform(const std::vector<float> &wav) {
  std::vector <std::vector<float>> feats;
  std::vector<float> waves;
  waves.insert(waves.end(), remained_wav_.begin(), remained_wav_.end());
  waves.insert(waves.end(), wav.begin(), wav.end());
  int num_frames = fbank_.Compute(waves, &feats);
  for (auto const& it : feats) {
    auto *this_feature = new std::vector<float>(it);
    if (!config_.cmvn_file.empty())
      ApplyCmvn(this_feature);
    features_.PushBack(this_feature);
  }
  num_frames_ += num_frames;

  unsigned int left_samples = waves.size() - config_.frame_shift * num_frames;
  remained_wav_.resize(left_samples);
  std::copy(waves.begin() + config_.frame_shift * num_frames, waves.end(),
            remained_wav_.begin());
}

void FeaturePipeline::SetInputFinished() {
  CHECK(!input_finished_);
  {
    input_finished_ = true;
  }
}

bool FeaturePipeline::GetFrame(int frame, std::vector<float> *feat) {
  auto base_feat = features_.At(frame);
  feat->clear();
  feat->assign(base_feat->begin(), base_feat->end());
  return true;
}

void FeaturePipeline::Reset() {
  input_finished_ = false;
  num_frames_ = 0;
  remained_wav_.clear();
  features_.Clear();
}

bool FeaturePipeline::LoadCmvn(const std::string& cmvn_file) {
  std::ifstream cmvn(cmvn_file, std::ios::in);
  if (!cmvn.is_open()) {
    LOG(FATAL) << "cmvn file open failed";
    return false;
  }
  std::string dummy1, dummy2;
  cmvn >> dummy1 >> dummy2;
  CHECK_EQ(dummy1, "global");
  CHECK_EQ(dummy2, "[");
  float m, v;
  means_.clear();
  istd_.clear();
  for (int i = 0; i < feature_dim_; i++) {
    cmvn >> m;
    means_.push_back(m);
  }
  for (int i = 0; i < feature_dim_; i++) {
    cmvn >> v;
    istd_.push_back(1.0 / sqrt(v));
  }
  cmvn >> dummy1;
  CHECK_EQ(dummy1, "]");

  return true;
}

bool FeaturePipeline::ApplyCmvn(std::vector<float> *feats) {
  CHECK_EQ(feats->size(), means_.size());
  CHECK_EQ(feats->size(), istd_.size());
  CHECK_EQ(feats->size(), feature_dim_);
  for (int i = 0; i < feats->size(); i++) {
    (*feats)[i] = (*feats)[i] - means_[i];
    (*feats)[i] = (*feats)[i] * istd_[i];
  }
  return true;
}

} // namespace athena