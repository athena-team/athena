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

#ifndef UTILS_UTILS_H_
#define UTILS_UTILS_H_

#include <cstdint>
#include <limits>
#include <vector>

namespace athena {

#define ATHENA_DISALLOW_COPY_AND_ASSIGN(Type) \
  Type(const Type &) = delete;                \
  Type &operator=(const Type &) = delete;

const float kFloatMax = std::numeric_limits<float>::max();

// Return the sum of two probabilities in log scale
float LogAdd(float x, float y);

template <typename T>
void TopK(const std::vector<T>& data,
          int32_t k,
          std::vector<T>* values,
          std::vector<int>* indices);


enum AthenaStatus {
  STATUS_OK      =  0,
  STATUS_ERROR   = -1,
};

}  // namespace athena

#endif  // UTILS_UTILS_H_