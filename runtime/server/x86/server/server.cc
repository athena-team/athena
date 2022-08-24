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

#include "server.h"

#include <hv/json.hpp>
#include <hv/base64.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace athena {

using Json = nlohmann::json;

enum EhandlerError { FAILEDTOPARSEINPUT, FAILEDTOPROCESSINPUT };

athena::DecoderResourcePtr resource;

bool InitGlobalResource(athena::DecodeType type) {
  resource = NewDecoderResourcePtr(type);
  if (nullptr == resource) {
    return false;
  }
  return true;
}

static std::string GetHostname() {
  static std::string g_hostname{};
  if (g_hostname.empty()) {
    char name[1024] = { 0 };
    if (auto ret = gethostname(name, sizeof(name)); ret == 0) {
      g_hostname = std::string(name);
    } else {
      g_hostname = "unknown";
    }
  }
  return g_hostname;
}

std::atomic< int > SessionHandler::counter{ 0 };
SessionHandler::SessionHandler() {
  counter.fetch_add(1);
  boost::uuids::uuid a_uuid = boost::uuids::random_generator()();
  std::string reqid  = boost::uuids::to_string(a_uuid);
  traceid_ = "default_" + reqid;
};

SessionHandler::~SessionHandler() {
  counter.fetch_sub(1);
};

bool SessionHandler::OnSessionOpen() {
  context_ = NewContextInterfacePtr(resource);
  if (nullptr == context_) {
    LOG(INFO) << "init decoder context failed";
    return false;
  }
  context_->Reset();
  LOG(INFO) << "init decoder context finished";
  return true;
}

std::string SessionHandler::OnReceiveMessage(const std::string& msg) {
  Json respJ(Json::object()), profileJ(Json::object());
  std::string audio;
  bool speech_end;
  auto root = Json::parse(msg);
  if(root.find("audio") == root.end() || root.find("speech_end") == root.end()) {
    LOG(INFO) << "OnReceiveMessage json failed: " << msg;
    errcode_ = FAILEDTOPARSEINPUT;
    goto finish;
  } else {
    audio = hv::Base64Decode(root["audio"].get<std::string>().c_str(),
                             root["audio"].get<std::string>().size());
    speech_end = root["speech_end"].get<bool>();
  }

  if (not initialized) {
    if (not OnSessionOpen()) {
      LOG(INFO) << "load decoder resource failed. traceid=" << traceid_;
      goto finish;
    }
    initialized = true;
    LOG(INFO) << "load decoder resource success. traceid=" << traceid_;
  }

  {
    if (not context_->InvokeDecoder(audio.c_str(), audio.size(), speech_end)) {
      LOG(INFO) << "process failed. traceid: " << traceid_;
      errcode_ = FAILEDTOPROCESSINPUT;
    } else {
      LOG(INFO) << "process success. traceid: " << traceid_;
    }
    respJ["payload"] = SerializeResult(speech_end);
    goto finish;
  }

  finish:
  profileJ["hostname"]    = GetHostname();
  profileJ["concurrency"] = counter.load();
  respJ["code"]           = errcode_;
  respJ["traceid"]        = traceid_;
  respJ["profile"]        = profileJ;
  return respJ.dump();
}

void SessionHandler::OnSessionClose() {
  LOG(INFO) << "OnSessionClose traceid: " << traceid_;
}

std::string SessionHandler::SerializeResult(bool finish) {
  Json nbestJ(Json::array());
  for (const auto& one_best : context_->GetResult()) {
    Json one_bestJ(Json::object());
    one_bestJ["sentence"] = one_best.sentence;
    if (finish) {
      Json word_piecesJ(Json::array());
      for (const auto& word_piece : one_best.word_pieces) {
        Json word_pieceJ(Json::object());
        word_pieceJ["word"] = word_piece.word;
        word_pieceJ["start"] = word_piece.start;
        word_pieceJ["end"] = word_piece.end;
        word_piecesJ.emplace_back(word_pieceJ);
      }
      one_bestJ["word_pieces"] = word_piecesJ;
    }
    nbestJ.emplace_back(one_bestJ);
    if (nbestJ.size() == nbest_) {
      break;
    }
  }
  return nbestJ.dump();
}

std::string SessionHandler::GetTraceid() {
  return traceid_;
}

}  // namespace athena