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

#ifndef DECODER_ITF_H_
#define DECODER_ITF_H_

#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <limits>

namespace athena {

enum DecodeType {
  kEnd2End = 0x00,
  kWfstBeamSearch = 0x01,
  kPrefixBeamSearch = 0x02,
  kTriggeredAttention = 0x03,
};

class DecoderResource;
class ContextInterface;
using DecoderResourcePtr = std::shared_ptr< DecoderResource >;
using ContextInterfacePtr = std::shared_ptr< ContextInterface >;

struct WordPiece {
  std::string word;
  int start = 0;
  int end = 0;

  WordPiece(std::string word, int start, int end)
      : word(std::move(word)), start(start), end(end) {}
};

struct DecodeResult {
  float score = -std::numeric_limits<float>::max();
  std::string sentence;
  std::vector<WordPiece> word_pieces;

  static bool CompareFunc(const DecodeResult& a, const DecodeResult& b) {
    return a.score > b.score;
  }
};

class ContextInterface {
 public:
  virtual ~ContextInterface() = default;

  virtual bool InvokeDecoder(const char *data, size_t len, bool last_frame) = 0;
  virtual bool Rescoring() = 0;
  virtual void Reset() = 0;
  virtual std::vector<DecodeResult> GetResult() const = 0;

 public:
  std::vector<DecodeResult> decode_result_;
};

class DecoderResource {
 public:
  virtual ~DecoderResource() = default;
  virtual DecodeType Type() const = 0;
};

/* @API Initialization language and acoustic model */
DecoderResourcePtr NewDecoderResourcePtr(DecodeType);

/* @API Create context to save intermediate data */
ContextInterfacePtr NewContextInterfacePtr(const DecoderResourcePtr&);

}

#endif // DECODER_ITF_H_