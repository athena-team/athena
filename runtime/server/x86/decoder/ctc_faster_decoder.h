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

#ifndef DECODER_CTC_FASTER_DECODER_H_
#define DECODER_CTC_FASTER_DECODER_H_

#include <cassert>

#include "itf/faster_decoder_itf.h"
#include "decoder/ctc_decodable.h"

namespace athena {

class CTCFasterDecoder:public athena::FasterDecoderInterface {
 public:
  typedef athena::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  CTCFasterDecoder(const athena::StdVectorFst & fst,
                   const athena::FasterDecoderOptions & config);

  void SetOptions(const athena::FasterDecoderOptions & config) override {
    config_ = config;
  }

  ~CTCFasterDecoder() override {
    ClearToks(toks_.Clear());
  }

  void Decode(DecodableInterface * decodable) override;
  bool ReachedFinal() override;
  void InitDecoding() override;
  void AdvanceDecoding(DecodableInterface * decodable,
                       int32 max_num_frames = -1) override;
  int32 NumFramesDecoded() override {
    return num_frames_decoded_;
  }
  bool GetBestPath(std::vector<int>& trans);
  bool GetNBestPath(std::vector<std::pair<std::vector<int>, float>>& nbest_trans, int nbest);

 protected:
  class Token {
   public:
    Arc arc_;
    Token *prev_;
    int32 ref_count_;
    double cost_;

    inline Token(const Arc & arc, BaseFloat ac_cost,
                 Token * prev):arc_(arc), prev_(prev),
                               ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        cost_ =
            prev->cost_ + arc.weight.Value() +
                ac_cost;
      } else {
        cost_ = arc.weight.Value() + ac_cost;
      }
    }
    inline Token(const Arc & arc, Token * prev):arc_(arc),
                                                prev_(prev), ref_count_(1) {
      if (prev) {
        prev->ref_count_++;
        cost_ =
            prev->cost_ + arc.weight.Value();
      } else {
        cost_ = arc.weight.Value();
      }
    }
    inline bool operator <(const Token & other) const {
      return cost_ > other.cost_;
    }

    inline static void TokenDelete(Token * tok) {
      while (--tok->ref_count_ == 0) {
        Token *prev = tok->prev_;
        delete tok;
        if (prev == nullptr)
          return;
        else
          tok = prev;
      }
      assert(tok->ref_count_ > 0);
    }
  };
  class NbestToken {
   public:
    double cost_;
    Token * tok_;
    NbestToken(double cost, Token* tok) {
      this->cost_ = cost;
      this->tok_ = tok;
    }
    bool operator>(const NbestToken & other) const {
      if(this->tok_ == other.tok_)
        return false;
      return this->cost_ > other.cost_;
    }
  };

  typedef HashList < StateId, Token * >::Elem Elem;
  double GetCutoff(Elem * list_head, size_t * tok_count,
                   BaseFloat * adaptive_beam, Elem ** best_elem);
  void PossiblyResizeHash(size_t num_toks);
  double ProcessEmitting(DecodableInterface * decodable);
  void ProcessNonemitting(double cutoff);
  HashList < StateId, Token * >toks_;
  const athena::StdVectorFst & fst_;
  FasterDecoderOptions config_;
  std::vector < StateId > queue_;
  std::vector < BaseFloat > tmp_array_;
  int32 num_frames_decoded_;
  void ClearToks(Elem * list);
};

} // namespace athena

#endif  //  DECODER_CTC_FASTER_DECODER_H_