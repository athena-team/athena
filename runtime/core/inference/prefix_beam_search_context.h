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

#ifndef INFERENCE_PREFIX_BEAM_SEARCH_CONTEXT_H_
#define INFERENCE_PREFIX_BEAM_SEARCH_CONTEXT_H_

#include "cppflow/ops.h"
#include "cppflow/model.h"
#include "inference/resource.h"
#include "utils/utils.h"
#include "utils/timer.h"

namespace athena {

class PrefixBeamSearchContext: public ContextInterface {
 public:
  ~PrefixBeamSearchContext() override = default;
  AthenaStatus Init(const AMDataPtr& am, const LMDataPtr& lm,
                    const DecodeOptionsPtr& decode_options_ptr);
  bool Rescoring() override;
  void Reset() override;
  bool InvokeDecoder(const char *data, size_t len, bool last_frame) override;
  std::vector<DecodeResult> GetResult() const override {
    int nbest = decode_result_.size() < nbest_ ? static_cast<int>(decode_result_.size()) : nbest_;
    return std::vector<DecodeResult>(decode_result_.begin(), decode_result_.begin()+nbest);
  }

  int feature_frame_shift_in_ms() const {
    return feature_pipeline_ptr_->config().frame_shift * 1000 /
        feature_pipeline_ptr_->config().sample_rate;
  }
  float ComputeAttentionScore(const std::vector<int>& hyp,
                              const std::vector<float>& prob,
                              int eos);
  void AttentionRescoring(const std::vector<std::vector<int>>& hyps,
                          float reverse_weight,
                          std::vector<float>* rescoring_score);

 private:
  void BuildDecodeResult(bool finish);

 private:
  AMDataPtr am_ = nullptr;
  LMDataPtr lm_ = nullptr;
  DecodeOptionsPtr decode_options_ptr_ = nullptr;
  FeaturePipelinePtr feature_pipeline_ptr_ = nullptr;
  DecodableCTCPtr decodable_ctc_ptr_ = nullptr;
  CtcPrefixBeamSearchPtr ctc_prefix_beam_search_ptr_ = nullptr;
  int num_frames_decoded_ = 0;
  int nbest_ = 1;
  int feat_bins = 0;
  int left_context = 0;
};

} // namespace athena

#endif //INFERENCE_PREFIX_BEAM_SEARCH_CONTEXT_H_