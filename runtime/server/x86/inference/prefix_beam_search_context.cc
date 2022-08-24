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

#include "prefix_beam_search_context.h"

namespace athena {

AthenaStatus PrefixBeamSearchContext::Init(const AMDataPtr& am, const LMDataPtr& lm,
                                           const DecodeOptionsPtr& decode_options_ptr) {
  if (nullptr == am || nullptr == lm || nullptr == decode_options_ptr) {
    LOG(ERROR) << "am lm or feature_config is null.";
    return STATUS_ERROR;
  }
  lm_ = lm; am_ = am;
  decode_options_ptr_ = decode_options_ptr;
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
  ctc_prefix_beam_search_ptr_.reset(new CtcPrefixBeamSearch(decode_options_ptr->ctc_prefix_beam_search_opts));
  if (nullptr == ctc_prefix_beam_search_ptr_){
    LOG(ERROR) << "init decoder failed.";
    return STATUS_ERROR;
  }
  feat_bins = decode_options_ptr->feature_pipeline_config.num_bins;
  left_context = decode_options_ptr->left_context;
  nbest_ = decode_options_ptr->nbest;
  return STATUS_OK;
}

void PrefixBeamSearchContext::Reset() {
  feature_pipeline_ptr_->Reset();
  decode_result_.clear();
  if(nullptr != decodable_ctc_ptr_)
    decodable_ctc_ptr_->Reset();
  if(nullptr != ctc_prefix_beam_search_ptr_)
    ctc_prefix_beam_search_ptr_->Reset();
  num_frames_decoded_ = 0;
}

bool PrefixBeamSearchContext::InvokeDecoder(const char *data, size_t len, bool last_frame) {
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
  std::vector<float> feats;
  for(;num_frames_decoded_ < num_frames; ++num_frames_decoded_) {
    std::vector<float> feat;
    feature_pipeline_ptr_->GetFrame(num_frames_decoded_, &feat);
    feats.insert(feats.end(), feat.begin(), feat.end());
  }
  unsigned int frames_decode = feats.size() / feat_bins;
  std::vector<cppflow::tensor> output_tensor =
      am_->GetAmNnet()->operator()({std::tuple<std::string, cppflow::tensor>("serving_default_x", cppflow::tensor(feats, {1, frames_decode, feat_bins, 1}))},
                                   {"StatefulPartitionedCall:0", "StatefulPartitionedCall:1"});
  std::vector<float> logits = output_tensor[0].get_data<float>();
  std::vector<float> encoder_output = output_tensor[1].get_data<float>();
  decodable_ctc_ptr_->Encode(logits, encoder_output);
  ctc_prefix_beam_search_ptr_->Search(dynamic_cast<DecodableInterface*>(decodable_ctc_ptr_.get()));
  BuildDecodeResult(last_frame);
  return true;
}

void PrefixBeamSearchContext::BuildDecodeResult(bool finish) {
  decode_result_.clear();
  auto words = lm_->GetWords();
  auto units = lm_->GetUnits();
  const auto& hypotheses = ctc_prefix_beam_search_ptr_->Outputs();
  const auto& inputs = ctc_prefix_beam_search_ptr_->Inputs();
  const auto& likelihood = ctc_prefix_beam_search_ptr_->Likelihood();
  const auto& times = ctc_prefix_beam_search_ptr_->Times();
  CHECK_EQ(hypotheses.size(), likelihood.size());
  for (size_t i = 0; i < hypotheses.size(); i++) {
    const std::vector<int>& hypothesis = hypotheses[i];

    DecodeResult result;
    result.score = likelihood[i];
    int offset = num_frames_decoded_ * feature_frame_shift_in_ms();
    for (auto it : hypothesis) {
      result.sentence += words[it];
    }

    // TimeStamp is only supported in final result
    // TimeStamp of the output of CtcWfstBeamSearch may be inaccurate due to
    // various FST operations when building the decoding graph. So here we use
    // time stamp of the input(e2e model unit), which is more accurate, and it
    // requires the symbol table of the e2e model used in training.

    if (!units.empty() && finish) {
      const std::vector<int>& input = inputs[i];
      const std::vector<int>& time_stamp = times[i];
      CHECK_EQ(input.size(), time_stamp.size());
      for (size_t j = 0; j < input.size(); j++) {
        std::string word = units[input[j]];
        int start = j > 0 ? ((time_stamp[j - 1] + time_stamp[j]) / 2 *
            feature_frame_shift_in_ms())
                          : 0;
        int end = j < input.size() - 1 ?
                  ((time_stamp[j] + time_stamp[j + 1]) / 2 * feature_frame_shift_in_ms()) :
                  num_frames_decoded_ * feature_frame_shift_in_ms();
        WordPiece word_piece(word, offset + start, offset + end);
        result.word_pieces.emplace_back(word_piece);
      }
    }

    decode_result_.emplace_back(result);
  }
}

bool PrefixBeamSearchContext::Rescoring() {
  // No need to do rescoring
  if (0.0 == decode_options_ptr_->rescoring_weight) {
    return true;
  }
  // Inputs() returns N-best input ids, which is the basic unit for rescoring
  // In CtcPrefixBeamSearch, inputs are the same to outputs
  const auto& hypotheses = ctc_prefix_beam_search_ptr_->Inputs();
  int num_hyps = hypotheses.size();
  if (num_hyps <= 0) {
    return false;
  }

  std::vector<float> rescoring_score;
  AttentionRescoring(
      hypotheses,
      decode_options_ptr_->reverse_weight,
      &rescoring_score);

  // Combine ctc score and rescoring score
  for (size_t i = 0; i < num_hyps; ++i) {
    decode_result_[i].score = decode_options_ptr_->rescoring_weight * rescoring_score[i] +
        decode_options_ptr_->ctc_weight * decode_result_[i].score;
  }
  std::sort(decode_result_.begin(), decode_result_.end(), DecodeResult::CompareFunc);
  return true;
}

float PrefixBeamSearchContext::ComputeAttentionScore(const std::vector<int>& hyp,
                                                     const std::vector<float>& prob,
                                                     int eos) {
  float score = 0.0f;
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += prob[j * am_->GetCtcOutputSize() + hyp[j]];
  }
  score += prob[hyp.size() * am_->GetCtcOutputSize() + eos];
  return score;
}

void PrefixBeamSearchContext::AttentionRescoring(
    const std::vector<std::vector<int>>& hyps,
    float reverse_weight,
    std::vector<float>* rescoring_score) {
  CHECK(rescoring_score != nullptr);
  int num_hyps = hyps.size();
  rescoring_score->resize(num_hyps, 0.0f);
  if (num_hyps == 0) {
    return;
  }
  // No encoder output
  std::vector<std::vector<float> > encoder_outputs_matrix = decodable_ctc_ptr_->GetEncoderOutputs();
  if (encoder_outputs_matrix.size() == 0) {
    return;
  }

  // Step 1: Prepare input
  int total_frames = encoder_outputs_matrix.size();
  int encoder_dim = encoder_outputs_matrix[0].size();
  std::vector<float> encoder_outputs;
  for(auto it : encoder_outputs_matrix) {
    encoder_outputs.insert(encoder_outputs.end(), it.begin(), it.end());
  }
  std::vector<int> hyps_length(num_hyps, 0);
  int max_hyps_len = 0;
  for (size_t i = 0; i < num_hyps; ++i) {
    int length = hyps[i].size() + 1;
    max_hyps_len = std::max(length, max_hyps_len);
    hyps_length[i] = static_cast<int64_t>(length);
  }
  std::vector<int> hyps_padding(num_hyps * max_hyps_len, 0);
  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    hyps_padding[i * max_hyps_len] = lm_->sos_;
    for (size_t j = 0; j < hyp.size(); ++j) {
      hyps_padding[i * max_hyps_len + j + 1] = hyp[j];
    }
  }

  // Step 2: Forward attention decoder by hyps and corresponding encoder_outs_
  std::tuple<std::string, cppflow::tensor>
      hyps_tensor("serving_default_hyps_pad1", cppflow::tensor(hyps_padding, {num_hyps, max_hyps_len}));
  std::tuple<std::string, cppflow::tensor>
      encoder_output_tensor("serving_default_encoder_outs",
                            cppflow::tensor(encoder_outputs, {1, total_frames, encoder_dim}));
  std::vector<std::tuple<std::string, cppflow::tensor> >
      inputs({hyps_tensor, encoder_output_tensor});
  std::vector<std::string> outputs({"StatefulPartitionedCall"});
  std::vector<cppflow::tensor> output_tensor =
      am_->GetRescoringNnet()->operator()(inputs, outputs);

  auto probs = output_tensor[0].get_data<float>();
  CHECK_EQ(probs.size(), num_hyps * max_hyps_len * am_->GetCtcOutputSize());
  std::vector<std::vector<float> > v_probs;
  for (int i = 0; i < num_hyps; ++i) {
    auto probs_first = probs.begin() + i * max_hyps_len * am_->GetCtcOutputSize();
    auto probs_second = probs.begin() + (i + 1) * max_hyps_len * am_->GetCtcOutputSize();
    v_probs.emplace_back(std::vector<float>(probs_first, probs_second));
  }

  // Step 3: Compute rescoring score
  for (size_t i = 0; i < num_hyps; ++i) {
    const std::vector<int>& hyp = hyps[i];
    float score = 0.0f;
    // left-to-right decoder score
    score = ComputeAttentionScore(hyp, v_probs[i], lm_->eos_);
    // Optional: Used for right to left score
    float r_score = 0.0f;
    // combined left-to-right and right-to-left score
    (*rescoring_score)[i] =  score * (1 - reverse_weight) +
        r_score * reverse_weight;
  }
}

} // namespace athena