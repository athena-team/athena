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

#ifndef DECODER_DECODABLE_ITF_H_
#define DECODER_DECODABLE_ITF_H_

#include <iostream>
#include <vector>

#include "utils/types.h"

namespace athena {

class DecodableInterface {
 public:
  virtual ~DecodableInterface() = default;
  virtual BaseFloat LogLikelihood(int32 frame, int32 index) = 0;
  virtual bool IsLastFrame(int32 frame) const = 0;
  virtual int32 NumFramesReady() const  = 0;
  virtual int32 NumIndices() const = 0;
  virtual std::vector<std::vector<float> > GetLogits() const = 0;
  virtual std::vector<std::vector<float> > GetEncoderOutputs() const = 0;
};

}  // namespace athena

#endif  // DECODER_DECODABLE_ITF_H_