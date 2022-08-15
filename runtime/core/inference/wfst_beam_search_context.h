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

#ifndef INFERENCE_WFST_BEAM_SEARCH_CONTEXT_H_
#define INFERENCE_WFST_BEAM_SEARCH_CONTEXT_H_

#include "cppflow/ops.h"
#include "cppflow/model.h"
#include "inference/resource.h"
#include "utils/utils.h"
#include "utils/timer.h"

namespace athena {

class WfstBeamSearchContext: public ContextInterface {
 public:
  ~WfstBeamSearchContext() override = default;
  AthenaStatus Init(const AMDataPtr& am, const LMDataPtr& lm,
                    const DecodeOptionsPtr& decode_options_ptr);
  bool Rescoring() override {
    LOG(ERROR) << "Rescoring is not supported in this decoder";
    return false;
  }
  void Reset() override;
  bool InvokeDecoder(const char *data, size_t len, bool last_frame) override;
  std::vector<DecodeResult> GetResult() const override {
    return decode_result_;
  }

  int feature_frame_shift_in_ms() const {
    return feature_pipeline_ptr_->config().frame_shift * 1000 /
        feature_pipeline_ptr_->config().sample_rate;
  }

 private:
  void BuildDecodeResult(std::vector<std::pair<std::vector<int>, float>>& predictions);

 private:
  AMDataPtr am_ = nullptr;
  LMDataPtr lm_ = nullptr;
  FeaturePipelinePtr feature_pipeline_ptr_ = nullptr;
  DecodableCTCPtr decodable_ctc_ptr_ = nullptr;
  CTCFasterDecoderPtr ctc_faster_decoder_ptr_ = nullptr;
  int num_frames_decoded_ = 0;
  int nbest_ = 1;
  int feat_bins = 0;
  int left_context = 0;
  cppflow::tensor inference_cache1_;
  cppflow::tensor inference_cache2_;
  std::string inference_cache1_name_;
  std::string inference_cache2_name_;
};

} // namespace athena

#endif //INFERENCE_WFST_BEAM_SEARCH_CONTEXT_H_