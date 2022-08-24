#!/bin/bash
# coding=utf-8
# Copyright (C) ATHENA AUTHORS
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# source some path
. ./tools/env.sh

# rnn language model

tail -n +2 examples/asr/hkust/data/train.csv |\
awk '{print $3"\t"$3}' > examples/asr/hkust/data/train.trans.csv
tail -n +2 examples/asr/hkust/data/dev.csv |\
awk '{print $3"\t"$3}' > examples/asr/hkust/data/dev.trans.csv
python athena/main.py examples/asr/hkust/configs/rnnlm.json || exit 1

echo "RNNLM language model is OK!! "
