// Copyright (C) 2022 ATHENA DECODER AUTHORS; Yang Han; Rui Yan
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

#ifndef DECODER_FASTER_DECODER_ITF_H_
#define DECODER_FASTER_DECODER_ITF_H_

#include <cassert>

#include "utils/hash_list.h"
#include "decodable_itf.h"
#include "fst/athena_fst.h"

namespace athena {

struct FasterDecoderOptions {
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat beam_delta;
  BaseFloat hash_ratio;
  FasterDecoderOptions():beam(16.0), max_active(std::numeric_limits < int32 >::max()),
                         min_active(20), beam_delta(0.5), hash_ratio(2.0) {}
};

class FasterDecoderInterface {
 public:
  virtual ~FasterDecoderInterface() = default;
  virtual void SetOptions(const FasterDecoderOptions & config) = 0;
  virtual void Decode(DecodableInterface * decodable) = 0;
  virtual bool ReachedFinal() = 0;
  virtual void InitDecoding() = 0;
  virtual void AdvanceDecoding(DecodableInterface * decodable,
                               int32 max_num_frames) = 0;
  virtual int32 NumFramesDecoded() = 0;
};

} // namespace athena

#endif  //DECODER_FASTER_DECODER_ITF_H_