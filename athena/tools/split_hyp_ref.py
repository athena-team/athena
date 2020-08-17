# coding=utf-8
# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
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
import os
import re
import argparse

def main(args):
    decode_f_dir = args.decode_result
    vocab_f_dir = args.vocab_dir
    base_dir = os.path.dirname(decode_f_dir)

    with open(decode_f_dir, "r", encoding="utf-8") as f:
        lines = f.readlines()
    refs = []
    hyps = []
    for i in lines:
        hyp, ref = i.split(']], shape')
        hyp = re.findall(r"\d+", hyp)
        ref = re.findall(r"\d+", ref.split('labels')[1].split('err')[0])
        refs.append([int(k) for k in ref])
        hyps.append([int(k) for k in hyp])

    with open(vocab_f_dir, "r", encoding="utf-8") as f:
        k_v = f.readlines()
    vocab_d = {}
    for i in k_v:
        _k = int(i.split(' ')[1].strip())
        _v = i.split(' ')[0]
        vocab_d[_k] = _v

    with open(os.path.join(base_dir, 'hyp.txt'), 'w', encoding="utf-8") as f:
        n = 0
        for line in hyps:
            n += 1
            c = ''
            for i in line[:-1]:
                c = c + vocab_d[i] + ' '
            f.write(c[:-1] + '(' + str(n) + '.wav)\n')
    with open(os.path.join(base_dir, 'ref.txt'), "w", encoding="utf-8") as f:
        n = 0
        for line in refs:
            n += 1
            c = ''
            for i in line:
                c = c + vocab_d[i] + ' '
            f.write(c[:-1] + '(' + str(n) + ".wav)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode_result', default='')
    parser.add_argument('--vocab_dir', default='')
    args = parser.parse_args()
    main(args)
