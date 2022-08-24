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

#include <hv/WebSocketClient.h>
#include <hv/base64.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "frontend/audio_pack_sim.h"

using namespace hv;

DEFINE_bool(simulate_streaming, false, "simulate streaming input");
DEFINE_double(chunk_length_secs, 0.4, "duration of a voice block");
DEFINE_string(url, "ws://127.0.0.1:8888", "url of websocket server");
DEFINE_int32(nbest, 1, "n-best of decode result");
DEFINE_string(wav_path, "", "test wav file path");

void ReadLoopFunc(WebSocketClient& ws) {
  bool final_chunk = false;
  athena::AudioPackageSimulator audio_pack_sim;
  audio_pack_sim.Init(FLAGS_simulate_streaming, static_cast<float>(FLAGS_chunk_length_secs), FLAGS_wav_path);
  while (!final_chunk) {
    std::vector<int16_t> input_data;
    if (!audio_pack_sim.GetData(input_data, final_chunk)) {
      LOG(ERROR) << "Did not find audio for utterance " << FLAGS_wav_path;
    }
    std::string b64_input_data =
        Base64Encode((unsigned char *) input_data.data(), input_data.size() * sizeof(int16_t));
    Json send_data {
        { "audio", b64_input_data},
        { "speech_end", final_chunk},
    };
    ws.send(send_data.dump());
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  WebSocketClient ws;
  std::thread t(ReadLoopFunc, std::ref(ws));
  ws.onopen = [&ws]() {
  };
  ws.onclose   = []() {
    LOG(INFO) << "onclose";
  };
  ws.onmessage = [](const std::string& msg) {
    LOG(INFO) << "onmessage: " << msg;
  };

  ReconnectInfo reconn;
  reconn.min_delay = 1000;
  reconn.max_delay = 10000;
  reconn.delay_policy = 2;
  ws.setReconnect(&reconn);

  ws.open(FLAGS_url.c_str());
  t.join();

  // press Enter to stop
  while (getchar() != '\n');
  return 0;
}