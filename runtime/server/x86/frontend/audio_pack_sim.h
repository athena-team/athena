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

#ifndef ATHENA_AUDIO_PACK_SIM_H_
#define ATHENA_AUDIO_PACK_SIM_H_

#include <vector>
#include <unistd.h>
#include <thread>

#include "frontend/wav.h"
#include "utils/timer.h"

namespace athena {

class AudioPackageSimulator {
 public:
  /* Initialize this class with audiopackagesimulator conf and wave data */
  bool Init(bool audio_sim, float chunk_length_secs, const std::string& wave_data) {
    do_audio_stream_sim_ = audio_sim;
    wave_data_.Init(wave_data);
    sample_rate_ = wave_data_.sample_rate();
    num_sample_ = wave_data_.num_sample();

    if (chunk_length_secs > 0) {
      chunk_length_ = std::min(max_chunk_length_, static_cast<int>(static_cast<float>(sample_rate_) * chunk_length_secs));
      if (chunk_length_ == 0) chunk_length_ = static_cast<int>(static_cast<float>(sample_rate_) * 0.1);
    } else {
      chunk_length_ = max_chunk_length_;
    }

    if (sample_rate_ == 16000 || sample_rate_ == 8000){
      data_ = wave_data_.data();
      wave_data_len_ = data_.size();
      time_per_samp_ms_ =  1000.f / static_cast<float>(sample_rate_);
    } else {
      return false;
    }
    return true;
  }

  int sample_rate() const { return sample_rate_; }

  int num_sample() const { return num_sample_; }

  /* get part of the wave data to process */
  bool GetData(std::vector<int16_t> &part_data, bool &final_chunk){
    int num_samp;
    /* You can not get more data if samp_offset_ equals to or larger than wave_data_len */
    if (samp_offset_ >= wave_data_len_) {
      return false;
    } else if (samp_offset_ < wave_data_len_) {
      /* Add delay to simulate ream-time input */
      int sim_elapsed = timer_.Elapsed();
      int wait_time = std::max(static_cast<int>(static_cast<float>(samp_offset_) * time_per_samp_ms_) - sim_elapsed,
                               last_chunk_ms_ - sim_elapsed + last_timestamp_ms_);
      if (wait_time > 0) {
        LOG(INFO) << "Simulate streaming, waiting for " << wait_time << "ms";
        std::this_thread::sleep_for (
            std::chrono::milliseconds(static_cast<int>(wait_time)));
      }

      int samp_remaining = wave_data_len_ - samp_offset_;
      num_samp = chunk_length_ < samp_remaining ? chunk_length_
                                                : samp_remaining;
      last_timestamp_ms_ = sim_elapsed;
      last_chunk_ms_ = static_cast<int>(static_cast<float>(num_samp) * time_per_samp_ms_);

      /* final chunk */
      if (num_samp == samp_remaining) {
        final_chunk = true;
      } else {
        final_chunk = false;
      }
      if (!do_audio_stream_sim_) {
        samp_offset_ = 0;
        num_samp = wave_data_len_;
        final_chunk = true;
      }

      std::vector<float> wave_part(data_.begin()+samp_offset_, data_.begin()+samp_offset_+num_samp);
      for (auto it : wave_part) {
        part_data.push_back(it);
      }
      samp_offset_ += num_samp;
    }
    return true;
  }

  ~AudioPackageSimulator() = default;

 private:
  WavReader wave_data_;
  Timer timer_;
  std::vector<float> data_;
  int sample_rate_;
  int num_sample_;
  int wave_data_len_;
  float time_per_samp_ms_;
  int samp_offset_ = 0;
  bool do_audio_stream_sim_ = false; /* if false, return the whole wave data */
  int max_chunk_length_ = 32000;
  int chunk_length_ = 0;
  int last_timestamp_ms_ = 0;
  int last_chunk_ms_ = 0;
};
} // namespace athena

#endif // ATHENA_AUDIO_PACK_SIM_H_
