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

#ifndef FRONTEND_FEATURE_PIPELINE_H_
#define FRONTEND_FEATURE_PIPELINE_H_

#include <string>
#include <vector>
#include <memory>

#include <glog/logging.h>

#include "frontend/fbank.h"
#include "frontend/recycling_vector.h"

namespace athena {

struct FeaturePipelineConfig {
  int num_bins;
  int sample_rate;
  int frame_length;
  int frame_shift;
  std::string cmvn_file;
  FeaturePipelineConfig(int num_bins, int sample_rate, std::string cmvn_file="")
      : num_bins(num_bins),         // 80 dim fbank
        sample_rate(sample_rate),
        cmvn_file(std::move(cmvn_file)) {  // 16k sample rate
    frame_length = sample_rate / 1000 * 25;  // frame length 25ms
    frame_shift = sample_rate / 1000 * 10;   // frame shift 10ms
  }

  void Info() const {
    LOG(INFO) << "feature pipeline config"
              << " num_bins " << num_bins << " frame_length " << frame_length
              << "frame_shift" << frame_shift;
  }
};

// Typically, FeaturePipeline is used in two threads: one thread call
// AcceptWaveform() to add raw wav data, another thread call Read() to read
// feature. so it is important to make it thread safe, and the Read() call
// should be a blocking call when there is no feature in feature_queue_ while
// the input is not finished.

class FeaturePipeline {
 public:
  explicit FeaturePipeline(const FeaturePipelineConfig& config);
  ~FeaturePipeline(){
    Reset();
  }

  void AcceptWaveform(const std::vector<float>& wav);
  int FeatureDim() const { return feature_dim_; }
  const FeaturePipelineConfig& config() const { return config_; }
  void SetInputFinished();

  bool GetFrame(int frame, std::vector<float>* feat);

  void Reset();
  bool IsLastFrame(int frame) const {
    return input_finished_ && (frame == num_frames_ - 1);
  }
  int NumFramesReady() const {return num_frames_;}
  bool LoadCmvn(const std::string& cmvn_file);
  bool ApplyCmvn(std::vector<float>* feats);

 private:
  const FeaturePipelineConfig& config_;
  int feature_dim_;
  Fbank fbank_;
  RecyclingVector features_;
  int num_frames_;
  bool input_finished_;
  std::vector<float> remained_wav_;
  std::vector<float> means_;
  std::vector<float> istd_;// inverse of standard deviation
};

typedef std::shared_ptr<FeaturePipelineConfig> FeaturePipelineConfigPtr;
typedef std::shared_ptr<FeaturePipeline> FeaturePipelinePtr;

} // namespace athena

#endif  // FRONTEND_FEATURE_PIPELINE_H_