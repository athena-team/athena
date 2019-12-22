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
""" segment word """

import codecs
import sys
import re
from absl import logging
import jieba


def segment_trans(vocab_file, text_file):
    ''' segment transcripts according to vocab
        using Maximum Matching Algorithm
    Args:
      vocab_file: vocab file
      text_file: transcripts file
    Returns:
      seg_trans: segment words
    '''
    jieba.set_dictionary(vocab_file)
    with open(text_file, "r", encoding="utf-8") as text:
        lines = text.readlines()
        sents = ''
        for line in lines:
            seg_line = jieba.cut(line.strip(), HMM=False)
            seg_line = ' '.join(seg_line)
            sents += seg_line + '\n'
        return sents


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    if len(sys.argv) < 3:
        logging.warning('Usage: python {} vocab_file text_file'.format(sys.argv[0]))
        sys.exit()
    print(segment_trans(sys.argv[1], sys.argv[2]))
