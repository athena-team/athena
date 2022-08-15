// Copyright (C) 2019 ATHENA DECODER AUTHORS; Xiangang Li; Yang Han
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

#ifndef DECODER_GRAPH_IO_H_
#define DECODER_GRAPH_IO_H_

#include <fstream>
#include <iostream>
#include <cmath>

#include "athena_fst.h"

namespace athena {

static const int32 kFstMagicNumber = 2125659606;

std::istream &ReadTypeIO(std::istream &strm, std::string *s);

std::istream &ReadTypeIO(std::istream &strm, int32 &number);

std::istream &ReadTypeIO(std::istream &strm, int64 &number);

std::istream &ReadTypeIO(std::istream &strm, uint64 &number);

std::istream &ReadTypeIO(std::istream &strm, float &number);

athena::StdVectorFst *ReadGraph(const std::string& filename);

void WriteGraph(athena::StdVectorFst *pfst);

} // namespace athena

#endif //DECODER_GRAPH_IO_H_