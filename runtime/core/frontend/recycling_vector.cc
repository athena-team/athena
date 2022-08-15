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

#include "frontend/recycling_vector.h"
#include <glog/logging.h>

namespace athena {

RecyclingVector::RecyclingVector(int items_to_hold):
    items_to_hold_(items_to_hold == 0 ? -1 : items_to_hold),
    first_available_index_(0) {
}

RecyclingVector::~RecyclingVector() {
  for (auto *item : items_) {
    delete item;
  }
}

std::vector<float> *RecyclingVector::At(int index) const {
  if (index < first_available_index_) {
    LOG(FATAL) << "Attempted to retrieve feature vector that was "
                  "already removed by the RecyclingVector (index = "
               << index << "; "
               << "first_available_index = " << first_available_index_ << "; "
               << "size = " << Size() << ")";
  }
  // 'at' does size checking.
  return items_.at(index - first_available_index_);
}

void RecyclingVector::PushBack(std::vector<float> *item) {
  if (items_.size() == items_to_hold_) {
    delete items_.front();
    items_.pop_front();
    ++first_available_index_;
  }
  items_.push_back(item);
}

int RecyclingVector::Size() const {
  return first_available_index_ + items_.size();
}

void RecyclingVector::Clear() {
  for (auto *item : items_) {
    delete item;
  }

  items_.clear();
  first_available_index_ = 0;
}

} // namespace athena