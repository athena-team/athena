// Copyright (C) 2022 ATHENA DECODER AUTHORS; Yang Han
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

#ifndef FRONTEND_RECYCLING_VECTOR_H_
#define FRONTEND_RECYCLING_VECTOR_H_

#include <vector>
#include <deque>

/// This class serves as a storage for feature vectors with an option to limit
/// the memory usage by removing old elements. The deleted frames indices are
/// "remembered" so that regardless of the MAX_ITEMS setting, the user always
/// provides the indices as if no deletion was being performed.
/// This is useful when processing very long recordings which would otherwise
/// cause the memory to eventually blow up when the features are not being removed.
namespace athena {

class RecyclingVector {

 public:
  /// By default it does not remove any elements.
  RecyclingVector(int items_to_hold = -1);

  /// The ownership is being retained by this collection - do not delete the item.
  std::vector<float> *At(int index) const;

  /// The ownership of the item is passed to this collection - do not delete the item.
  void PushBack(std::vector<float> *item);

  /// This method returns the size as if no "recycling" had happened,
  /// i.e. equivalent to the number of times the PushBack method has been called.
  int Size() const;

  void Clear();

  ~RecyclingVector();

 private:
  std::deque<std::vector<float>*> items_;
  int items_to_hold_;
  int first_available_index_;
};

} // namespace athena

#endif  // FRONTEND_RECYCLING_VECTOR_H_