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

#include "wfst_beam_search_context.h"

namespace athena {

AthenaStatus WfstBeamSearchContext::Init(const AMDataPtr& am, const LMDataPtr& lm,
                                         const DecodeOptionsPtr& decode_options_ptr) {
  if (nullptr == am || nullptr == lm || nullptr == decode_options_ptr) {
    LOG(ERROR) << "am lm or feature_config is null.";
    return STATUS_ERROR;
  }
  lm_ = lm; am_ = am;
  feature_pipeline_ptr_.reset(new FeaturePipeline(decode_options_ptr->feature_pipeline_config));
  if (nullptr == feature_pipeline_ptr_) {
    LOG(ERROR) << "feature pipeline init failed.";
    return STATUS_ERROR;
  }
  decodable_ctc_ptr_.reset(new DecodableCTC(am_->GetCtcOutputSize(), decode_options_ptr->ctc_decodable_options));
  if (nullptr == decodable_ctc_ptr_){
    LOG(ERROR) << "init decodable failed.";
    return STATUS_ERROR;
  }
  ctc_faster_decoder_ptr_.reset(new CTCFasterDecoder(*lm_->GetGraph(), decode_options_ptr->faster_decoder_options));
  if (nullptr == ctc_faster_decoder_ptr_){
    LOG(ERROR) << "init decoder failed.";
    return STATUS_ERROR;
  }
  ctc_faster_decoder_ptr_->InitDecoding();
  feat_bins = decode_options_ptr->feature_pipeline_config.num_bins;
  left_context = decode_options_ptr->left_context;
  nbest_ = decode_options_ptr->nbest;
  inference_cache1_name_ = decode_options_ptr->model_options.inference_cache1_name;
  inference_cache2_name_ = decode_options_ptr->model_options.inference_cache2_name;
  inference_cache1_ = cppflow::tensor(std::vector<float>(am_->GetCache1Size(),0), am_->GetCache1Shape());
  inference_cache2_ =  cppflow::tensor(std::vector<float>(am_->GetCache2Size(),0), am_->GetCache2Shape());
  return STATUS_OK;
}

void WfstBeamSearchContext::Reset() {
  feature_pipeline_ptr_->Reset();
  decode_result_.clear();
  if(nullptr != decodable_ctc_ptr_)
    decodable_ctc_ptr_->Reset();
  if(nullptr != ctc_faster_decoder_ptr_)
    ctc_faster_decoder_ptr_->InitDecoding();
  num_frames_decoded_ = 0;
  inference_cache1_ = cppflow::tensor(std::vector<float>(am_->GetCache1Size(),0), am_->GetCache1Shape());
  inference_cache2_ = cppflow::tensor(std::vector<float>(am_->GetCache2Size(),0), am_->GetCache2Shape());
}

bool WfstBeamSearchContext::InvokeDecoder(const char *data, size_t len, bool last_frame) {
  if(len < 0 || nullptr == data){
    LOG(ERROR) << "Recieved wrong samples";
    return false;
  }
  std::vector<int16_t> wav_data(reinterpret_cast<const int16_t*>(data),
                                reinterpret_cast<const int16_t*>(data) + len / sizeof(int16_t));
  feature_pipeline_ptr_->AcceptWaveform( std::vector<float> (wav_data.begin(), wav_data.end()));
  int num_frames = feature_pipeline_ptr_->NumFramesReady();
  if(last_frame) {
    feature_pipeline_ptr_->SetInputFinished();
  }
  std::vector<float> feats, his_feats;
  if(left_context > 0) {
    int num_frames_copy = std::min(left_context, num_frames_decoded_);
    for(int num_frame=num_frames_decoded_ - num_frames_copy; num_frame < num_frames_decoded_; ++num_frame) {
      std::vector<float> feat;
      feature_pipeline_ptr_->GetFrame(num_frame, &feat);
      his_feats.insert(his_feats.end(), feat.begin(), feat.end());
    }
  }
  for(;num_frames_decoded_ < num_frames; ++num_frames_decoded_) {
    std::vector<float> feat;
    feature_pipeline_ptr_->GetFrame(num_frames_decoded_, &feat);
    feats.insert(feats.end(), feat.begin(), feat.end());
  }
  unsigned int frames_decode = feats.size() / feat_bins;
  unsigned int his_frames_decode = his_feats.size() / feat_bins;
  std::tuple<std::string, cppflow::tensor>
      input_tensor("serving_default_x", cppflow::tensor(feats, {1, frames_decode, feat_bins, 1}));
  std::tuple<std::string, cppflow::tensor>
      inference_cache1_tensor(inference_cache1_name_, cppflow::tensor(his_feats, {1, his_frames_decode, feat_bins, 1}));
  std::tuple<std::string, cppflow::tensor>
      inference_cache2_tensor(inference_cache2_name_, inference_cache2_);
  std::vector<std::tuple<std::string, cppflow::tensor> >
      inputs({input_tensor, inference_cache1_tensor, inference_cache2_tensor});
  std::vector<std::string>
      outputs({"StatefulPartitionedCall:0", "StatefulPartitionedCall:1", "StatefulPartitionedCall:2"});
  std::vector<cppflow::tensor> output_tensor =
      am_->GetAmNnet()->operator()(inputs, outputs);
  std::vector<float> logits = output_tensor[0].get_data<float>();
  inference_cache2_ = output_tensor[2];
  decodable_ctc_ptr_->Encode(logits);
  ctc_faster_decoder_ptr_->AdvanceDecoding(dynamic_cast<DecodableInterface*>(decodable_ctc_ptr_.get()));
  std::vector<std::pair<std::vector<int>, float>> predictions;
  ctc_faster_decoder_ptr_->GetNBestPath(predictions, nbest_);
  BuildDecodeResult(predictions);
  return true;
}

void WfstBeamSearchContext::BuildDecodeResult(std::vector<std::pair<std::vector<int>, float>>& predictions) {
  decode_result_.clear();
  auto words = lm_->GetWords();
  for (auto const& it : predictions) {
    DecodeResult result;
    for(const auto& id : it.first){
      result.sentence += words[id];
    }
    result.score = it.second;
    decode_result_.emplace_back(result);
  }
}

} // namespace athena