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

#ifndef DECODER_CTC_DECODABLE_H_
#define DECODER_CTC_DECODABLE_H_

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <algorithm>

#include "itf/decodable_itf.h"

namespace athena{

struct encoder_output {
  std::vector<std::vector<float> > logits;
  std::vector<std::vector<float> > encoder_out;
  encoder_output() {
    logits.clear();
    encoder_out.clear();
  }
  void clear() {
    logits.clear();
    encoder_out.clear();
  }
  ~encoder_output() {
    logits.clear();
    encoder_out.clear();
  }
};

struct CTCDecodableOptions {
  int blank_id;
  bool ctc_prune;
  BaseFloat scale;
  BaseFloat minus_blank;
  BaseFloat ctc_threshold;
  BaseFloat prior_scale = 0.0;
  BaseFloat *prior_log_scores;
  CTCDecodableOptions():blank_id(0), ctc_prune(false), scale(3.0), minus_blank(0.0),
                        ctc_threshold(0.0), prior_scale(0.0), prior_log_scores(nullptr) {}
};

class DecodableCTC:public athena::DecodableInterface {
 public:
  DecodableCTC(int ctc_output_size, CTCDecodableOptions ctc_decodable_options);

  int Reset(); // call this function when decoder reset
  void Encode(std::vector<float> &logits);
  void Encode(std::vector<float> &logits, std::vector<float> &encoder_output);
  bool IsBlankFrame(int32 frame) const;
  BaseFloat LogLikelihood(int32 frame, int32 idx) override;

  int32 NumFramesReady() const override {
    return num_frames_calculated_;
  }
  bool IsLastFrame(int32 frame) const override {
    assert(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }
  bool IsCTCPrune() const {
    return this->ctc_prune_;
  }
  int32 NumIndices() const override {
    return ncol_;
  }
  std::vector<std::vector<float> > GetLogits() const override {
    return likes_.logits;
  };
  std::vector<std::vector<float> > GetEncoderOutputs() const override {
    return likes_.encoder_out;
  };

 private:
  encoder_output likes_;
  int nrow_;
  int ncol_;
  /*
  for stream input, decodable object must remember the total frames that have been
  calculated,and then get am scores from likes_ treating num_frames_calculated_ as offsets.
  Thus the decodable object should not be created more than one time for speech until the decoder reset.
  if there is only one decodable object for one decoder,then decodable object must call Reset() when decoder call Reset()
  */
  int ctc_output_size_;
  int num_frames_calculated_;
  BaseFloat scale_;
  int blank_id_;
  BaseFloat minus_blank_;
  bool ctc_prune_;
  BaseFloat ctc_threshold_;
  BaseFloat *prior_log_scores_ = nullptr;
  BaseFloat prior_scale_;
};

} // namespace athena

#endif  // DECODER_CTC_DECODABLE_H_