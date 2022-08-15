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

#include "decoder/ctc_faster_decoder.h"

#include <cassert>
#include <set>
#include <queue>

#include "utils/types.h"

namespace athena {

CTCFasterDecoder::CTCFasterDecoder(const athena::StdVectorFst &fst,
                                   const FasterDecoderOptions &opts):
    fst_(fst), config_(opts), num_frames_decoded_(-1) {
  assert(config_.hash_ratio >= 1.0);  // less doesn't make much sense.
  assert(config_.max_active > 1);
  assert(config_.min_active >= 0 && config_.min_active < config_.max_active);
  toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
}

void CTCFasterDecoder::InitDecoding() {
  // clean up from last time:
  ClearToks(toks_.Clear());
  StateId start_state = fst_.Start();
  assert(start_state != -1);
  Arc dummy_arc(0, 0, Weight::One(), start_state);
  toks_.Insert(start_state, new Token(dummy_arc, nullptr));
  ProcessNonemitting(std::numeric_limits<float>::max());
  num_frames_decoded_ = 0;
}

void CTCFasterDecoder::Decode(DecodableInterface *decodable) {
  auto decodablectc = dynamic_cast<DecodableCTC*>(decodable);
  InitDecoding();
  while (!decodablectc->IsLastFrame(num_frames_decoded_ - 1)) {
    if(decodablectc->IsCTCPrune() && decodablectc->IsBlankFrame(num_frames_decoded_)){
      num_frames_decoded_++;
      continue;
    }
    double weight_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(weight_cutoff);
  }
}

void CTCFasterDecoder::AdvanceDecoding(DecodableInterface *decodable,
                                       int32 max_num_frames) {
  auto decodablectc = dynamic_cast<DecodableCTC*>(decodable);
  assert(num_frames_decoded_ >= 0 &&
      "You must call InitDecoding() before AdvanceDecoding()");
  int32 num_frames_ready = decodablectc->NumFramesReady();
  assert(num_frames_ready >= num_frames_decoded_);
  int32 target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0)
    target_frames_decoded = std::min(target_frames_decoded,
                                     num_frames_decoded_ + max_num_frames);
  while (num_frames_decoded_ < target_frames_decoded) {
    // note: ProcessEmitting() increments num_frames_decoded_
    if(decodablectc->IsCTCPrune() && decodablectc->IsBlankFrame(num_frames_decoded_)){
      num_frames_decoded_++;
      continue;
    }
    double weight_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(weight_cutoff);
  }
}

bool CTCFasterDecoder::ReachedFinal() {
  for (const Elem *e = toks_.GetList(); e != nullptr; e = e->tail) {
    if (e->val->cost_ != std::numeric_limits<double>::infinity() &&
        fst_.Final(e->key).Value() != Weight::Zero().Value())
      return true;
  }
  return false;
}

bool CTCFasterDecoder::GetBestPath(std::vector<int>& trans) {
  trans.clear();
  Token *best_tok = nullptr;
  bool is_final = ReachedFinal();
  if (!is_final) {
    for (const Elem *e = toks_.GetList(); e != nullptr; e = e->tail)
      if (best_tok == nullptr || *best_tok < *(e->val))
        best_tok = e->val;
  } else {
    double infinity =  std::numeric_limits<double>::infinity(),
        best_cost = infinity;
    for (const Elem *e = toks_.GetList(); e != nullptr; e = e->tail) {
      double this_cost = e->val->cost_ + fst_.Final(e->key).Value();
      if (this_cost < best_cost && this_cost != infinity) {
        best_cost = this_cost;
        best_tok = e->val;
      }
    }
  }
  if (best_tok == nullptr) return false;  // No output.
  for (Token *tok = best_tok; tok != nullptr; tok = tok->prev_) {
    if(tok->arc_.olabel != 0){
      trans.push_back(tok->arc_.olabel);
    }
  }
  std::reverse(trans.begin(),trans.end());
  return true;
}

bool CTCFasterDecoder::GetNBestPath(std::vector<std::pair<std::vector<int>, float>>& nbest_trans, int nbest) {
nbest_trans.clear();
std::priority_queue<NbestToken,
vector<NbestToken>,
std::greater<NbestToken> > nbest_queue;
bool is_final = ReachedFinal();
if (!is_final) {
Token *best_tok = nullptr;
for (const Elem *e = toks_.GetList(); e != nullptr; e = e->tail) {
if (best_tok == nullptr || *best_tok < *(e->val))
best_tok = e->val;
}
if (best_tok != nullptr) {
nbest_queue.push(NbestToken(best_tok->cost_, best_tok));
}
} else {
double infinity =  std::numeric_limits<double>::infinity(),
    best_cost = infinity;
for (const Elem *e = toks_.GetList(); e != nullptr; e = e->tail) {
double this_cost = e->val->cost_ + fst_.Final(e->key).Value();
if (this_cost != infinity) {
nbest_queue.push(NbestToken(this_cost, e->val));
}
}
}
if (nbest_queue.empty()) return false;  // No output.
for (int i = 0; i < nbest; ++i) {
std::vector<int> trans;
for (Token *tok = nbest_queue.top().tok_; tok != nullptr; tok = tok->prev_) {
if(tok->arc_.olabel != 0) {
trans.emplace_back(tok->arc_.olabel);
}
}
std::reverse(trans.begin(),trans.end());
std::pair<std::vector<int>, float> output = std::make_pair(trans, nbest_queue.top().cost_);
nbest_trans.emplace_back(output);
nbest_queue.pop();
}
return true;
}


// Gets the weight cutoff.  Also counts the active tokens.
double CTCFasterDecoder::GetCutoff(Elem *list_head, size_t *tok_count,
                                   BaseFloat *adaptive_beam, Elem **best_elem) {
  double best_cost = std::numeric_limits<double>::infinity();
  size_t count = 0;
  if (config_.max_active == std::numeric_limits<int32>::max() &&
      config_.min_active == 0) {
    for (Elem *e = list_head; e != nullptr; e = e->tail, count++) {
      double w = e->val->cost_;
      if (w < best_cost) {
        best_cost = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != nullptr) *tok_count = count;
    if (adaptive_beam != nullptr) *adaptive_beam = config_.beam;
    return best_cost + config_.beam;
  } else {
    tmp_array_.clear();
    for (Elem *e = list_head; e != nullptr; e = e->tail, count++) {
      double w = e->val->cost_;
      tmp_array_.push_back(w);
      if (w < best_cost) {
        best_cost = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != nullptr) *tok_count = count;
    double beam_cutoff = best_cost + config_.beam,
        min_active_cutoff = std::numeric_limits<double>::infinity(),
        max_active_cutoff = std::numeric_limits<double>::infinity();

    if (tmp_array_.size() > static_cast<size_t>(config_.max_active)) {
      std::nth_element(tmp_array_.begin(),
                       tmp_array_.begin() + config_.max_active,
                       tmp_array_.end());
      max_active_cutoff = tmp_array_[config_.max_active];
    }
    if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
      if (adaptive_beam)
        *adaptive_beam = max_active_cutoff - best_cost + config_.beam_delta;
      return max_active_cutoff;
    }
    if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
      if (config_.min_active == 0) min_active_cutoff = best_cost;
      else {
        std::nth_element(tmp_array_.begin(),
                         tmp_array_.begin() + config_.min_active,
                         tmp_array_.size() > static_cast<size_t>(config_.max_active) ?
                         tmp_array_.begin() + config_.max_active :
                         tmp_array_.end());
        min_active_cutoff = tmp_array_[config_.min_active];
      }
    }
    if (min_active_cutoff > beam_cutoff) { // min_active is looser than beam.
      if (adaptive_beam)
        *adaptive_beam = min_active_cutoff - best_cost + config_.beam_delta;
      return min_active_cutoff;
    } else {
      *adaptive_beam = config_.beam;
      return beam_cutoff;
    }
  }
}

void CTCFasterDecoder::PossiblyResizeHash(size_t num_toks) {
  auto new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
      * config_.hash_ratio);
  if (new_sz > toks_.Size()) {
    toks_.SetSize(new_sz);
  }
}

// ProcessEmitting returns the likelihood cutoff used.
double CTCFasterDecoder::ProcessEmitting(DecodableInterface *decodable) {
  auto decodablectc = dynamic_cast<DecodableCTC*>(decodable);
  int32 frame = num_frames_decoded_;
  Elem *last_toks = toks_.Clear();
  size_t tok_cnt;
  BaseFloat adaptive_beam;
  Elem *best_elem = nullptr;
  double weight_cutoff = GetCutoff(last_toks, &tok_cnt,
                                   &adaptive_beam, &best_elem);
  PossiblyResizeHash(tok_cnt);  // This makes sure the hash is always big enough.
  double next_weight_cutoff = std::numeric_limits<double>::infinity();
  if (best_elem) {
    StateId state = best_elem->key;
    Token *tok = best_elem->val;
    for (athena::ArcIterator aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) {  // we'd propagate..
        BaseFloat ac_cost = - decodablectc->LogLikelihood(frame, arc.ilabel);
        double new_weight = arc.weight.Value() + tok->cost_ + ac_cost;
        if (new_weight + adaptive_beam < next_weight_cutoff)
          next_weight_cutoff = new_weight + adaptive_beam;
      }
    }
  }

  for (Elem *e = last_toks, *e_tail; e != nullptr; e = e_tail) {  // loop this way
    // because we delete "e" as we go.
    StateId state = e->key;
    Token *tok = e->val;
    if (tok->cost_ < weight_cutoff) {  // not pruned.
      assert(state == tok->arc_.nextstate);
      for (athena::ArcIterator aiter(fst_, state);
           !aiter.Done();
           aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          BaseFloat ac_cost =  - decodablectc->LogLikelihood(frame, arc.ilabel);
          double new_weight = arc.weight.Value() + tok->cost_ + ac_cost;
          if (new_weight < next_weight_cutoff) {  // not pruned..
            auto *new_tok = new Token(arc, ac_cost, tok);
            Elem *e_found = toks_.Find(arc.nextstate);
            if (new_weight + adaptive_beam < next_weight_cutoff)
              next_weight_cutoff = new_weight + adaptive_beam;
            if (e_found == nullptr) {
              toks_.Insert(arc.nextstate, new_tok);
            } else {
              if ( *(e_found->val) < *new_tok ) {
                Token::TokenDelete(e_found->val);
                e_found->val = new_tok;
              } else {
                Token::TokenDelete(new_tok);
              }
            }
          }
        }
      }
    }
    e_tail = e->tail;
    Token::TokenDelete(e->val);
    toks_.Delete(e);
  }
  num_frames_decoded_++;
  return next_weight_cutoff;
}

// TODO: first time we go through this, could avoid using the queue.
void CTCFasterDecoder::ProcessNonemitting(double cutoff) {
  // Processes nonemitting arcs for one frame.
  assert(queue_.empty());
  for (const Elem *e = toks_.GetList(); e != nullptr;  e = e->tail)
    queue_.push_back(e->key);
  while (!queue_.empty()) {
    StateId state = queue_.back();
    queue_.pop_back();
    Token *tok = toks_.Find(state)->val;  // would segfault if state not
    if (tok->cost_ > cutoff) { // Don't bother processing successors.
      continue;
    }
    assert(tok != nullptr && state == tok->arc_.nextstate);
    for (athena::ArcIterator aiter(fst_, state);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        auto *new_tok = new Token(arc, tok);
        if (new_tok->cost_ > cutoff) {  // prune
          Token::TokenDelete(new_tok);
        } else {
          Elem *e_found = toks_.Find(arc.nextstate);
          if (e_found == nullptr) {
            toks_.Insert(arc.nextstate, new_tok);
            queue_.push_back(arc.nextstate);
          } else {
            if ( *(e_found->val) < *new_tok ) {
              Token::TokenDelete(e_found->val);
              e_found->val = new_tok;
              queue_.push_back(arc.nextstate);
            } else {
              Token::TokenDelete(new_tok);
            }
          }
        }
      }
    }
  }
}

void CTCFasterDecoder::ClearToks(Elem *list) {
  for (Elem *e = list, *e_tail; e != nullptr; e = e_tail) {
    Token::TokenDelete(e->val);
    e_tail = e->tail;
    toks_.Delete(e);
  }
}

} // namespace athena