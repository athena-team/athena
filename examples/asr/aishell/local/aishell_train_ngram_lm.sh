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

lm_config=$1  #examples/asr/aishell/data/5gram.arpa

if  [ ! -x "$(command -v ./tools/kenlm/build/bin/lmplz)" ]; then
        echo "lmplz is not installed !! "
		echo "Checkout tools/install_kenlm.sh, run bash tools/install_sclite.sh to install"
        exit 1
fi

# 4gram language model

tail -n +2 examples/asr/aishell/data/train.csv |\
cut -f 3 > examples/asr/aishell/data/text
python examples/asr/aishell/local/segment_word.py \
        examples/asr/aishell/data/vocab \
        examples/asr/aishell/data/text \
        > examples/asr/aishell/data/text.seg || exit 1
tools/kenlm/build/bin/lmplz -o 5 < examples/asr/aishell/data/text.seg \
         > $lm_config || exit 1

echo "4gram language model is OK!! "
