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

#ifndef SERVER_HANDLER_H_
#define SERVER_HANDLER_H_

#include <atomic>
#include <deque>
#include <memory>
#include <string>
#include <tuple>

#include <glog/logging.h>

#include "include/decoder_itf.h"

namespace athena {
bool InitGlobalResource(athena::DecodeType type);

class SessionHandler {
 public:
  SessionHandler();
  ~SessionHandler();

  bool OnSessionOpen();
  std::string OnReceiveMessage(const std::string& msg);
  void OnSessionClose();

  std::string SerializeResult(bool finish);
  std::string GetTraceid();

 private:
  static std::atomic< int > counter;
  ContextInterfacePtr context_ = nullptr;
  std::string traceid_;
  bool initialized = false;
  int nbest_ = 1;
  int errcode_ = 0;
};
using PSessionHandler = std::shared_ptr< SessionHandler >;

}  // namespace athena

#endif //SERVER_HANDLER_H_