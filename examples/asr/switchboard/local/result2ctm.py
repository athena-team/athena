# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS
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
# Only support tensorflow 2.0
# pylint: disable=invalid-name, no-member
r""" a script for switchboard dataset to transform decode result into ctm file """
import sys
import re
from absl import logging
import sentencepiece as spm


def reformat_number(number):
    # e.g. reformat number "0" into "0.00"
    num = str(int(number))
    decimal_num = (num[:-2] if len(num) > 2 else "0") + "." + (num[-2:] if len(num) > 1 else "00")
    return decimal_num


def main(decode_list_file, utt_key_file, spm_model_file, ctm_file):
    # decode bpe pieces' ids into words and reformat results
    utt_keys = []
    decode_results = []
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model_file)
    ctm = open(ctm_file, "w", encoding="utf-8")

    with open(utt_key_file, "r", encoding="utf-8") as utt_key:
        lines = utt_key.readlines()
        for line in lines:
            utt_keys.append(line.strip())

    with open(decode_list_file, "r", encoding="utf-8") as decode_list:
        lines = decode_list.readlines()
        for line in lines:
            # modify "2000" if you use different bpe amount rather than 2000
            decode_result_ids = [int(item) for item in line.split("[[")[1].split("]]")[0].split() if item != "2000"]
            decode_result_words = sp.DecodeIds(decode_result_ids)
            decode_results.append(decode_result_words)

    assert len(utt_keys) == len(decode_results)
    for i in range(len(utt_keys)):
        key = utt_keys[i]
        hyp = decode_results[i]
        ctm_line = ""
        infos = re.split("[-_]", key)
        begin_time = float(infos[3])
        end_time = float(infos[4])
        hyp_split = hyp.split(" ")
        each_gap_time = (end_time - begin_time) / len(hyp_split)
        for count, item in enumerate(hyp_split):
            ctm_line = infos[0] + "_" + infos[1] + " " + infos[2]
            ctm_line += " " + reformat_number(begin_time)
            if count != len(hyp_split) - 1:
                ctm_line += " " + reformat_number(each_gap_time)
                begin_time += each_gap_time
            else:
                ctm_line += " " + reformat_number(end_time - begin_time)
            ctm_line += " " + item.upper()
            ctm.write(ctm_line + "\n")
    ctm.close()


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) != 5:
        logging.warning('Usage: python {} decode_list_file utt_key_file spm_model_file ctm_file(w)'.format(sys.argv[0]))
        sys.exit()
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
