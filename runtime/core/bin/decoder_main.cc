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

#include <iostream>
#include <fstream>
#include <iomanip>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "include/decoder_itf.h"
#include "frontend/wav.h"
#include "frontend/audio_pack_sim.h"
#include "utils/thread_pool.h"

DEFINE_bool(simulate_streaming, false, "simulate streaming input");
DEFINE_double(chunk_length_secs, 0.4, "duration of a voice block");
DEFINE_string(wav_scp, "", "input wav scp");
DEFINE_string(res_path, "", "result output file");
DEFINE_string(decoder_type, "End2End", "decode type, support: End2End, WfstBeamSearch, PrefixBeamSearch");
DEFINE_int32(thread_num, 1, "test thread num");

int total_waves_dur = 0;
class DecodeTask: public athena::Task {
 public:
  DecodeTask(const athena::DecoderResourcePtr& resource, std::string key,
             std::string path, std::string trans_path)
      :context_(NewContextInterfacePtr(resource)), key_(std::move(key)),
       path_(std::move(path)), trans_path_(std::move(trans_path)) {}

  void run() override {
    context_->Reset();
    bool final_chunk = false;
    athena::AudioPackageSimulator audio_pack_sim;
    audio_pack_sim.Init(FLAGS_simulate_streaming, static_cast<float>(FLAGS_chunk_length_secs), path_);
    while (!final_chunk) {
      std::vector<int16_t> input_data;
      if (!audio_pack_sim.GetData(input_data, final_chunk)) {
        LOG(ERROR) << "Did not find audio for utterance " << path_;
      }
      context_->InvokeDecoder((char*)input_data.data(), input_data.size() * sizeof(int16_t), final_chunk);
      std::string trans;
      if (!context_->GetResult().empty())
        trans = context_->GetResult()[0].sentence;
      if (!trans.empty())
        LOG(INFO) << "result: " << trans;
      if (final_chunk) {
        std::ofstream transfile(trans_path_, std::ios::app);
        transfile << key_ << "	" << trans << std::endl;
        transfile.close();
      }
    }
    int wave_dur =
        static_cast<int>(audio_pack_sim.num_sample() /
            audio_pack_sim.sample_rate() * 1000);
    total_waves_dur += wave_dur;
  }

 private:
  std::string key_;
  std::string path_;
  std::string trans_path_;
  athena::ContextInterfacePtr context_;
};

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  std::string waves_scp_path = FLAGS_wav_scp,
      trans_path = FLAGS_res_path;
  auto *tp = new athena::ThreadPool(FLAGS_thread_num);
  athena::DecoderResourcePtr resource;
  if (FLAGS_decoder_type == "End2End"){
    resource = athena::NewDecoderResourcePtr(athena::kEnd2End);
  } else if (FLAGS_decoder_type == "WfstBeamSearch") {
    resource = athena::NewDecoderResourcePtr(athena::kWfstBeamSearch);
  } else if (FLAGS_decoder_type == "PrefixBeamSearch") {
    resource = athena::NewDecoderResourcePtr(athena::kPrefixBeamSearch);
  } else {
    LOG(ERROR) << "The current decoder type is not supported";
    return 0;
  }
  if (nullptr == resource) {
    LOG(ERROR) << "Asr resource init fail";
    return 0;
  }
  std::ifstream wav_scp(waves_scp_path, std::ios::in);
  std::string key, path;
  athena::Timer timer;
  while(!wav_scp.eof()) {
    if(wav_scp>>key>>path) {
      tp->addTask(new DecodeTask(resource, key, path, trans_path));
    }
  }
  wav_scp.close();
  delete tp;
  int total_decode_time = timer.Elapsed();
  LOG(INFO) << "Total: decoded " << total_waves_dur << "ms audio taken "
            << total_decode_time << "ms.";
  LOG(INFO) << "RTF: " << std::setprecision(4)
            << static_cast<float>(total_decode_time) / static_cast<float>(total_waves_dur);
  return 0;
}