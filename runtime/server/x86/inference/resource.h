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

#ifndef INFERENCE_RESOURCE_H_
#define INFERENCE_RESOURCE_H_

#include "include/decoder_itf.h"
#include "inference/tensor_model.h"
#include "frontend/feature_pipeline.h"
#include "decoder/ctc_decodable.h"
#include "decoder/ctc_faster_decoder.h"
#include "decoder/ctc_prefix_beam_search.h"
#include "utils/utils.h"

namespace athena {

using FeaturePipelinePtr = std::shared_ptr< FeaturePipeline >;
using DecodableInterfacePtr = std::shared_ptr< DecodableInterface >;
using DecodableCTCPtr = std::shared_ptr< DecodableCTC >;
using CTCFasterDecoderPtr = std::shared_ptr< CTCFasterDecoder >;
using CtcPrefixBeamSearchPtr = std::shared_ptr< CtcPrefixBeamSearch >;

struct DecodeOptions {
  float ctc_weight = 0.0;
  float rescoring_weight = 1.0;
  float reverse_weight = 0.0;
  int nbest = 1;
  int left_context = 0;
  FeaturePipelineConfig feature_pipeline_config;
  ModelOptions model_options;
  CTCDecodableOptions ctc_decodable_options;
  CtcPrefixBeamSearchOptions ctc_prefix_beam_search_opts;
  FasterDecoderOptions faster_decoder_options;
  DecodeOptions():feature_pipeline_config(FeaturePipelineConfig(80, 16000)) {}
};
using DecodeOptionsPtr = std::shared_ptr< DecodeOptions >;

class E2EResource: public DecoderResource {
 public:
  explicit  E2EResource(const DecodeOptions& decode_options);
  ~E2EResource() override = default;

  DecodeType Type() const override {
    return kEnd2End;
  }

  AthenaStatus InitOptions();

  AMDataPtr GetAM() {
    return am_;
  }
  LMDataPtr GetLM() {
    return lm_;
  }
  DecodeOptionsPtr GetDecodeOptions() {
    return decode_options_ptr_;
  }

 private:
  AthenaStatus LoadModel();

 private:
  AMDataPtr am_ = nullptr;
  LMDataPtr lm_ = nullptr;
  DecodeOptionsPtr decode_options_ptr_;
};

class WfstBeamSearchResource: public DecoderResource {
 public:
  explicit WfstBeamSearchResource(const DecodeOptions& decode_options);
  ~WfstBeamSearchResource() override = default;

  DecodeType Type() const override {
    return kWfstBeamSearch;
  }

  AthenaStatus InitOptions();

  AMDataPtr GetAM() {
    return am_;
  }
  LMDataPtr GetLM() {
    return lm_;
  }
  DecodeOptionsPtr GetDecodeOptions() {
    return decode_options_ptr_;
  }

 private:
  AthenaStatus LoadModel();

 private:
  AMDataPtr am_ = nullptr;
  LMDataPtr lm_ = nullptr;
  DecodeOptionsPtr decode_options_ptr_;
};

class PrefixBeamSearchResource: public DecoderResource {
 public:
  explicit PrefixBeamSearchResource(const DecodeOptions& decode_options);
  ~PrefixBeamSearchResource() override = default;

  DecodeType Type() const override {
    return kPrefixBeamSearch;
  }

  AthenaStatus InitOptions();

  AMDataPtr GetAM() {
    return am_;
  }
  LMDataPtr GetLM() {
    return lm_;
  }
  DecodeOptionsPtr GetDecodeOptions() {
    return decode_options_ptr_;
  }

 private:
  AthenaStatus LoadModel();

 private:
  AMDataPtr am_ = nullptr;
  LMDataPtr lm_ = nullptr;
  DecodeOptionsPtr decode_options_ptr_;
};

} // namespace athena

#endif  // INFERENCE_RESOURCE_H_