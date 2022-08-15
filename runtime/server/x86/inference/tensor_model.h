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

#ifndef INFERENCE_TENSOR_MODEL_H_
#define INFERENCE_TENSOR_MODEL_H_

#include <string>
#include <memory>
#include <cassert>

#include <glog/logging.h>

#include "cppflow/model.h"
#include "fst/graph_io.h"
#include "utils/utils.h"

namespace athena {

struct ModelOptions {
  std::string am_path;
  std::string lm_path;
  std::string rescoring_path;
  std::string dict_path;
  std::string unit_path;
  std::string inference_cache1_name;
  std::string inference_cache2_name;
};

class AMData {
 public:
  explicit AMData(ModelOptions model_options):model_options_(std::move(model_options)) {}
  ~AMData() {
    delete am_nnet_;
    delete rescoring_nnet_;
  }

  AthenaStatus LoadModel() {
    am_nnet_ = new cppflow::model(model_options_.am_path);
    if (nullptr == am_nnet_) {
      LOG(ERROR) << "read acoustic model failed.";
      return STATUS_ERROR;
    }
    auto output_shape = am_nnet_->get_operation_shape("StatefulPartitionedCall");
    ctc_output_size_ = output_shape[output_shape.size()-1];
    if (!model_options_.inference_cache1_name.empty()) {
      inference_cache1_shape_ = am_nnet_->get_operation_shape(model_options_.inference_cache1_name);
      for (auto& it : inference_cache1_shape_) {
        if (it == -1)
          it = 1;
        inference_cache1_size_ *= it;
      }
    }
    if (!model_options_.inference_cache2_name.empty()) {
      inference_cache2_shape_ = am_nnet_->get_operation_shape(model_options_.inference_cache2_name);
      for (auto& it : inference_cache2_shape_) {
        if (it == -1)
          it = 1;
        inference_cache2_size_ *= it;
      }
    }
    if (!model_options_.rescoring_path.empty()) {
      rescoring_nnet_ = new cppflow::model(model_options_.rescoring_path);
    }
    LOG(INFO) << "finish loading acoustic model.";
    return STATUS_OK;
  }

  cppflow::model *GetAmNnet() {
    return am_nnet_;
  }

  cppflow::model *GetRescoringNnet() {
    return rescoring_nnet_;
  }

  int GetCtcOutputSize() const {
    return ctc_output_size_;
  }

  std::vector<int64_t> GetCache1Shape() const {
    return inference_cache1_shape_;
  }

  std::vector<int64_t> GetCache2Shape() const {
    return inference_cache2_shape_;
  }

  int GetCache1Size() const {
    return inference_cache1_size_;
  }

  int GetCache2Size() const {
    return inference_cache2_size_;
  }

 private:
  ModelOptions model_options_;
  cppflow::model *am_nnet_ = nullptr;
  cppflow::model *rescoring_nnet_ = nullptr;
  int ctc_output_size_ = 0;
  std::vector<int64_t> inference_cache1_shape_;
  std::vector<int64_t> inference_cache2_shape_;
  int inference_cache1_size_ = 1;
  int inference_cache2_size_ = 1;

 public:
  ATHENA_DISALLOW_COPY_AND_ASSIGN(AMData);
};
typedef std::shared_ptr<AMData> AMDataPtr;


class LMData {
 public:
  explicit LMData(ModelOptions model_options):model_options_(std::move(model_options)){}
  ~LMData() {
    delete pgraph_;
  }

  AthenaStatus LoadFst() {
    LOG(INFO) << "Start init wfst decoder.";
    pgraph_ = ReadGraph(model_options_.lm_path);
    if(nullptr == pgraph_){
      LOG(ERROR) << "Read language model failed.";
      return STATUS_ERROR;
    }
    LOG(INFO) << "Finish loading language model.";

    if (model_options_.dict_path.empty()) {
      LOG(ERROR) << "Words file is null";
      return STATUS_ERROR;
    }
    std::ifstream dict_file(model_options_.dict_path, std::ios::in);
    if(dict_file.get()==EOF) {
      LOG(ERROR) << "words file is empty";
      return STATUS_ERROR;
    }
    int idx;
    std::string word;
    while(!dict_file.eof()){
      dict_file >> word >> idx;
      words_.emplace_back(word);
    }
    dict_file.close();
    assert(words_.size() > 1);
    LOG(INFO) << "Finish loading words file.";
    if (!model_options_.unit_path.empty()) {
      std::ifstream unit_file(model_options_.unit_path, std::ios::in);
      if(unit_file.get()==EOF) {
        LOG(ERROR) << "words file is empty";
        return STATUS_ERROR;
      }
      std::string unit;
      while(!unit_file.eof()){
        unit_file >> unit >> idx;
        units_.emplace_back(unit);
      }
      unit_file.close();
      assert(units_.size() > 1);
      LOG(INFO) << "Finish loading units file.";
    }
    return STATUS_OK;
  }

  AthenaStatus LoadWords() {
    if (model_options_.dict_path.empty()) {
      LOG(ERROR) << "Words file is null";
      return STATUS_ERROR;
    }
    std::ifstream dict_file(model_options_.dict_path, std::ios::in);
    if(dict_file.get() == EOF) {
      LOG(ERROR) << "words file is empty";
      return STATUS_ERROR;
    }
    int idx;
    std::string word;
    while(!dict_file.eof()){
      dict_file >> word >> idx;
      words_.emplace_back(word);
    }
    dict_file.close();
    LOG(INFO) << "finish loading words file.";
    assert(words_.size() > 1);
    units_ = words_;
    eos_ = words_.size() - 1;
    sos_ = words_.size() - 1;
    return STATUS_OK;
  }

  StdVectorFst *GetGraph() {
    return pgraph_;
  }

  std::vector<std::string> &GetWords() {
    return words_;
  }

  std::vector<std::string> &GetUnits() {
    return units_;
  }

 public:
  unsigned int eos_ = 0;
  unsigned int sos_ = 0;

 private:
  ModelOptions model_options_;
  StdVectorFst *pgraph_ = nullptr;
  std::vector<std::string> words_;
  std::vector<std::string> units_;

 public:
  ATHENA_DISALLOW_COPY_AND_ASSIGN(LMData);
};
typedef std::shared_ptr<LMData> LMDataPtr;

}  //namespace athena

#endif //INFERENCE_TENSOR_MODEL_H_