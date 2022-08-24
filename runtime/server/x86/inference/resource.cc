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

#include "inference/resource.h"

namespace athena {

E2EResource::E2EResource(const DecodeOptions& decode_options)
    :decode_options_ptr_(std::make_shared<DecodeOptions>(decode_options)) {}

AthenaStatus E2EResource::InitOptions() {
  if(LoadModel() != STATUS_OK) {
    LOG(ERROR) << "Load('" << decode_options_ptr_->model_options.am_path << "','"
               << decode_options_ptr_->model_options.lm_path << "') failed.";
    return STATUS_ERROR;
  }
  return STATUS_OK;
}

AthenaStatus E2EResource::LoadModel() {
  am_ = std::make_shared<AMData>(decode_options_ptr_->model_options);
  lm_ = std::make_shared<LMData>(decode_options_ptr_->model_options);
  if(am_->LoadModel() != STATUS_OK || lm_->LoadWords()!= STATUS_OK)
    return STATUS_ERROR;
  return STATUS_OK;
}

WfstBeamSearchResource::WfstBeamSearchResource(const DecodeOptions& decode_options)
    :decode_options_ptr_(std::make_shared<DecodeOptions>(decode_options)) {}

AthenaStatus WfstBeamSearchResource::InitOptions() {
  if(LoadModel() != STATUS_OK) {
    LOG(ERROR) << "Load('" << decode_options_ptr_->model_options.am_path << "','"
               << decode_options_ptr_->model_options.lm_path << "') failed.";
    return STATUS_ERROR;
  }
  return STATUS_OK;
}

AthenaStatus WfstBeamSearchResource::LoadModel() {
  am_ = std::make_shared<AMData>(decode_options_ptr_->model_options);
  lm_ = std::make_shared<LMData>(decode_options_ptr_->model_options);
  if(am_->LoadModel() != STATUS_OK || lm_->LoadFst() != STATUS_OK)
    return STATUS_ERROR;
  return STATUS_OK;
}

PrefixBeamSearchResource::PrefixBeamSearchResource(const DecodeOptions& decode_options)
    :decode_options_ptr_(std::make_shared<DecodeOptions>(decode_options)) {}

AthenaStatus PrefixBeamSearchResource::InitOptions() {
  if(LoadModel() != STATUS_OK) {
    LOG(ERROR) << "Load('" << decode_options_ptr_->model_options.am_path << "','"
               << decode_options_ptr_->model_options.lm_path << "') failed.";
    return STATUS_ERROR;
  }
  return STATUS_OK;
}

AthenaStatus PrefixBeamSearchResource::LoadModel() {
  am_ = std::make_shared<AMData>(decode_options_ptr_->model_options);
  lm_ = std::make_shared<LMData>(decode_options_ptr_->model_options);
  if(am_->LoadModel() != STATUS_OK || lm_->LoadWords() != STATUS_OK)
    return STATUS_ERROR;
  return STATUS_OK;
}

} // namespace athena