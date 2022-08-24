// Copyright (C) 2022 ATHENA DECODER AUTHORS;  Yang Han; Rui Yan
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

#include "decoder/ctc_decodable.h"

namespace athena {

DecodableCTC::DecodableCTC(int ctc_output_size, CTCDecodableOptions ctc_decodable_options):
    ctc_output_size_(ctc_output_size), num_frames_calculated_(0), nrow_(0), ncol_(0) {
  scale_ = ctc_decodable_options.scale;
  blank_id_ = ctc_decodable_options.blank_id + 1;
  minus_blank_ = ctc_decodable_options.minus_blank;
  ctc_prune_ = ctc_decodable_options.ctc_prune;
  ctc_threshold_ = ctc_decodable_options.ctc_threshold;
  prior_log_scores_ = ctc_decodable_options.prior_log_scores;
  prior_scale_ = ctc_decodable_options.prior_scale;
  Reset();
}

int DecodableCTC::Reset() {
  num_frames_calculated_ = 0;
  likes_.clear();
  return 0;
}

void DecodableCTC::Encode(std::vector<float> &logits) {
  likes_.logits.clear();
  unsigned int frames_ready = logits.size() / ctc_output_size_;
  for (int i = 0; i < frames_ready; ++i) {
    auto logits_first = logits.begin() + i * ctc_output_size_;
    auto logits_second = logits.begin() + (i+1) * ctc_output_size_;
    likes_.logits.emplace_back(std::vector<float>(logits_first, logits_second));
  }
  nrow_ = likes_.logits.size();
  ncol_ = likes_.logits[0].size();
  num_frames_calculated_ += nrow_;
}

void DecodableCTC::Encode(std::vector<float> &logits, std::vector<float> &encoder_output) {
  likes_.logits.clear(); // only clear logits; encoder for rescoring
  unsigned int frames_ready = logits.size() / ctc_output_size_;
  unsigned int encoder_output_size = encoder_output.size() / frames_ready;
  for (int i = 0; i < frames_ready; ++i) {
    auto logits_first = logits.begin() + i * ctc_output_size_;
    auto logits_second = logits.begin() + (i+1) * ctc_output_size_;
    likes_.logits.emplace_back(std::vector<float>(logits_first, logits_second));
    auto encoder_first = encoder_output.begin() + i * encoder_output_size;
    auto encoder_second = encoder_output.begin() + (i+1) * encoder_output_size;
    likes_.encoder_out.emplace_back(std::vector<float>(encoder_first, encoder_second));
  }
  nrow_ = likes_.logits.size();
  ncol_ = likes_.logits[0].size();
  num_frames_calculated_ += nrow_;
}

bool DecodableCTC::IsBlankFrame(int32 frame) const {
  assert(frame < NumFramesReady());
  frame=frame-(num_frames_calculated_-nrow_);
  double blkscore=-likes_.logits[frame][blank_id_-1];
  if(blkscore < this->ctc_threshold_) {
    return true;
  }
  else {
    return false;
  }
}

BaseFloat DecodableCTC::LogLikelihood(int32 frame, int32 idx) {
  /*
   * return the index label's score in frame
   */
  assert(blank_id_ >= 1);
  assert(minus_blank_ >= 0);
  assert(idx-1 < ncol_);
  frame=frame-(num_frames_calculated_-nrow_);
  assert(frame < nrow_);

  double score;
  if(prior_log_scores_){
    score=likes_.logits[frame][idx-1] - (prior_scale_*prior_log_scores_[idx-1]);
  }else{
    score=likes_.logits[frame][idx-1];
  }
  if(idx==blank_id_){
    return scale_*(score-minus_blank_);
  }
  else{
    return scale_ * score;
  }
}

} // namespace athena