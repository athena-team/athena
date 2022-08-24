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

#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from absl import logging
import jieba

import kenlm


def seg_word2stdio(vocab_path, text_path):
    jieba.set_dictionary(vocab_path)  # "examples/asr/aishell/data/vocab"

    with open(text_path, "r") as fin:  # "examples/asr/aishell/data/text"
        for line in fin:
            seg_list = jieba.cut(line.strip(), cut_all=True, HMM=False)
            print(" ".join(seg_list))


def seg_word2file(vocab_path, text_path, file_out_path):
    jieba.set_dictionary(vocab_path)  # "examples/asr/aishell/data/vocab"

    with open(text_path, "r") as fin:  # "examples/asr/aishell/data/text"
        with open(file_out_path, "w") as fout:
            for line in fin:
                seg_list = jieba.cut(line.strip(), cut_all=True, HMM=False)
                fout.write(" ".join(seg_list)+"\n")


def test_lm():
    sentence = " ".join(list("预计第三季度将陆续有部分股市资金重归楼市"))
    words = ['<s>'] + sentence.split() + ['</s>']
    model = kenlm.LanguageModel("examples/asr/aishell/data/5gram.arpa")
    for i, (prob, length, oov) in enumerate(model.full_scores(sentence)):
        print('{0} {1}: {2}'.format(prob, length, ' '.join(words[i + 2 - length:i + 2])))
        if oov:
            print('\t"{0}" is an OOV'.format(words[i + 1]))

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        logging.info('Usage: python {} vocab_path text_path\n'
                     '    vocab_path : One word occupies one line; each line is divided into three parts:'
                     '    words, word frequency (omitted), entity (omitted)\n'
                     '    text_path  : text to segment'.format(sys.argv[0]))
        exit(1)
    vocab_path = sys.argv[1]
    text_path = sys.argv[2]
    if len(sys.argv) == 3:
        seg_word2stdio(vocab_path, text_path)
    else:
        file_out_path = sys.argv[3]
        seg_word2file(vocab_path, text_path, file_out_path)
