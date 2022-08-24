#ifndef DECODER_CTC_PREFIX_BEAM_SEARCH_H_
#define DECODER_CTC_PREFIX_BEAM_SEARCH_H_

#include <unordered_map>

#include "utils/utils.h"
#include "decoder/ctc_decodable.h"

namespace athena {

struct CtcPrefixBeamSearchOptions {
  int blank = 0;  // blank id
  int first_beam_size = 10;
  int second_beam_size = 10;
};

struct PrefixScore {
  float s = -kFloatMax;               // blank ending score
  float ns = -kFloatMax;              // none blank ending score
  float v_s = -kFloatMax;             // viterbi blank ending score
  float v_ns = -kFloatMax;            // viterbi none blank ending score
  float cur_token_prob = -kFloatMax;  // prob of current token
  std::vector<int> times_s;           // times of viterbi blank path
  std::vector<int> times_ns;          // times of viterbi none blank path

  float score() const { return LogAdd(s, ns); }
  float viterbi_score() const { return v_s > v_ns ? v_s : v_ns; }
  const std::vector<int>& times() const {
    return v_s > v_ns ? times_s : times_ns;
  }

  float total_score() const { return score(); }
};

struct PrefixHash {
  size_t operator()(const std::vector<int>& prefix) const {
    size_t hash_code = 0;
    // here we use KB&DR hash code
    for (int id : prefix) {
      hash_code = id + 31 * hash_code;
    }
    return hash_code;
  }
};

class CtcPrefixBeamSearch {
 public:
  explicit CtcPrefixBeamSearch(const CtcPrefixBeamSearchOptions& opts);

  void Search(DecodableInterface* decodable);
  void Reset();
  void UpdateHypotheses(
      const std::vector<std::pair<std::vector<int>, PrefixScore>>& hpys);

  const std::vector<float>& viterbi_likelihood() const {
    return viterbi_likelihood_;
  }
  const std::vector<std::vector<int>>& Inputs() const {
    return hypotheses_;
  }
  const std::vector<std::vector<int>>& Outputs() const {
    return hypotheses_;
  }
  const std::vector<float>& Likelihood() const { return likelihood_; }
  const std::vector<std::vector<int>>& Times() const { return times_; }

 private:
  int abs_time_step_ = 0;

  // N-best list and corresponding likelihood_, in sorted order
  std::vector<std::vector<int>> hypotheses_;
  std::vector<float> likelihood_;
  std::vector<float> viterbi_likelihood_;
  std::vector<std::vector<int>> times_;

  std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps_;
  std::vector<std::vector<int>> outputs_;
  const CtcPrefixBeamSearchOptions& opts_;

 public:
  ATHENA_DISALLOW_COPY_AND_ASSIGN(CtcPrefixBeamSearch);
};

} // namespace athena

#endif //DECODER_CTC_PREFIX_BEAM_SEARCH_H_