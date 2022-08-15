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

#include <gflags/gflags.h>

#include "include/decoder_itf.h"
#include "inference/resource.h"
#include "inference/e2e_context.h"
#include "inference/wfst_beam_search_context.h"
#include "inference/prefix_beam_search_context.h"

namespace athena {

// FeaturePipelineConfig flags
DEFINE_int32(num_bins, 80, "Num mel bins for fbank feature");
DEFINE_int32(sample_rate, 16000, "Sample rate for audio");
DEFINE_string(cmvn, "", "Exported cmvn file");

// Model flags
DEFINE_string(am_path, "", "Tensorflow exported acoustic model path");
DEFINE_string(rescoring_path, "", "Tensorflow exported rescoring model path");
DEFINE_string(lm_path, "", "Exported language model file");
DEFINE_string(dict_path, "", "Words symbol table path");
DEFINE_string(unit_path, "","e2e model unit symbol table, is used to get timestamp of the result");
DEFINE_int32(ctc_out_size, 0, "Output dimension of the model");
DEFINE_int32(left_context, 0, "Frames spliced on the left");

// DecodeOptions flags
DEFINE_double(acoustic_scale, 3.0, "Acoustic scale for ctc wfst search");
DEFINE_int32(blank_id, 0, "Id of blank in CTC");
DEFINE_double(minus_blank, 3.0, "Minus blank value for blank id");
DEFINE_bool(ctc_prune, false, "Whether prune for score of blank id");
DEFINE_double(ctc_threshold, 0.0, "Threshold of CTC prune");
DEFINE_double(beam, 10.0, "Beam in ctc wfst search");
DEFINE_int32(prefix_beam, 10, "Beam in ctc prefix search");
DEFINE_int32(max_active, 1000, "Max active states for one state");
DEFINE_int32(min_active, 20, "Min active states for one state");
DEFINE_int32(nbest, 1, "Nbest for ctc wfst");
DEFINE_string(inference_cache1_name, "", "model inference cache name.");
DEFINE_string(inference_cache2_name, "", "model inference cache name.");


DecodeOptions InitDecodeOptionsFromFlags() {
  DecodeOptions decode_options;
  decode_options.nbest = FLAGS_nbest;
  decode_options.left_context = FLAGS_left_context;
  decode_options.feature_pipeline_config =
      FeaturePipelineConfig(FLAGS_num_bins, FLAGS_sample_rate, FLAGS_cmvn);
  decode_options.model_options.am_path = FLAGS_am_path;
  decode_options.model_options.rescoring_path = FLAGS_rescoring_path;
  decode_options.model_options.lm_path = FLAGS_lm_path;
  decode_options.model_options.dict_path = FLAGS_dict_path;
  decode_options.model_options.unit_path = FLAGS_unit_path;
  decode_options.model_options.inference_cache1_name = FLAGS_inference_cache1_name;
  decode_options.model_options.inference_cache2_name = FLAGS_inference_cache2_name;
  decode_options.faster_decoder_options.beam = FLAGS_beam;
  decode_options.faster_decoder_options.max_active = FLAGS_max_active;
  decode_options.faster_decoder_options.min_active = FLAGS_min_active;
  decode_options.ctc_decodable_options.ctc_threshold = FLAGS_ctc_threshold;
  decode_options.ctc_decodable_options.ctc_prune = FLAGS_ctc_prune;
  decode_options.ctc_decodable_options.minus_blank = FLAGS_minus_blank;
  decode_options.ctc_decodable_options.blank_id = FLAGS_blank_id;
  decode_options.ctc_decodable_options.scale = FLAGS_acoustic_scale;
  decode_options.ctc_prefix_beam_search_opts.blank = FLAGS_blank_id;
  decode_options.ctc_prefix_beam_search_opts.first_beam_size = FLAGS_prefix_beam;
  decode_options.ctc_prefix_beam_search_opts.second_beam_size = FLAGS_prefix_beam;
  return decode_options;
}

DecoderResourcePtr NewDecoderResourcePtr(DecodeType type) {
  DecoderResourcePtr decoder_ptr;
  auto decode_options = InitDecodeOptionsFromFlags();
  if (type == kEnd2End) {
    decoder_ptr.reset(new E2EResource(decode_options));
    auto e2e_decoder_ptr = std::dynamic_pointer_cast<E2EResource>(decoder_ptr);
    if (e2e_decoder_ptr->InitOptions() != STATUS_OK) {
      LOG(ERROR) << "InitOptions failed";
      return nullptr;
    }
  }else if (type == kWfstBeamSearch) {
    decoder_ptr.reset(new WfstBeamSearchResource(decode_options));
    auto wfst_decoder_ptr = std::dynamic_pointer_cast<WfstBeamSearchResource>(decoder_ptr);
    if (wfst_decoder_ptr->InitOptions() != STATUS_OK) {
      LOG(ERROR) << "InitOptions failed";
      return nullptr;
    }
  }else if (type == kPrefixBeamSearch) {
    decoder_ptr.reset(new PrefixBeamSearchResource(decode_options));
    auto prefix_serach_ptr = std::dynamic_pointer_cast<PrefixBeamSearchResource>(decoder_ptr);
    if (prefix_serach_ptr->InitOptions() != STATUS_OK) {
      LOG(ERROR) << "InitOptions failed";
      return nullptr;
    }
  }else {
    LOG(ERROR) << "Decoder type wrong";
    return nullptr;
  }
  return decoder_ptr;
}

ContextInterfacePtr NewContextInterfacePtr(const DecoderResourcePtr& resource) {
  DecodeType type = resource->Type();
  ContextInterfacePtr context_ptr;
  if (type == kEnd2End) {
    auto resource_ptr = std::dynamic_pointer_cast<E2EResource>(resource);
    auto am = resource_ptr->GetAM();
    auto lm = resource_ptr->GetLM();
    auto decode_options = resource_ptr->GetDecodeOptions();
    context_ptr.reset(new E2EContext());
    auto e2e_context_ptr =
        std::dynamic_pointer_cast<E2EContext>(context_ptr);
    if (e2e_context_ptr->Init(am, lm, decode_options) != STATUS_OK) {
      LOG(ERROR) << "Init decoder context failed";
      return nullptr;
    }
  } else if (type == kWfstBeamSearch) {
    auto resource_ptr = std::dynamic_pointer_cast<WfstBeamSearchResource>(resource);
    auto am = resource_ptr->GetAM();
    auto lm = resource_ptr->GetLM();
    auto decode_options = resource_ptr->GetDecodeOptions();
    context_ptr.reset(new WfstBeamSearchContext());
    auto wfst_context_ptr =
        std::dynamic_pointer_cast<WfstBeamSearchContext>(context_ptr);
    if (wfst_context_ptr->Init(am, lm, decode_options) != STATUS_OK) {
      LOG(ERROR) << "Init decoder context failed";
      return nullptr;
    }
  } else if (type == kPrefixBeamSearch) {
    auto resource_ptr = std::dynamic_pointer_cast<PrefixBeamSearchResource>(resource);
    auto am = resource_ptr->GetAM();
    auto lm = resource_ptr->GetLM();
    auto decode_options = resource_ptr->GetDecodeOptions();
    context_ptr.reset(new PrefixBeamSearchContext());
    auto prefix_search_context_ptr =
        std::dynamic_pointer_cast<PrefixBeamSearchContext>(context_ptr);
    if (prefix_search_context_ptr->Init(am, lm, decode_options) != STATUS_OK) {
      LOG(ERROR) << "Init decoder context failed";
      return nullptr;
    }
  } else {
    LOG(ERROR) << "not support resource type";
    return nullptr;
  }
  return context_ptr;
}

} // namespace athena