# coding=utf-8
# Copyright (C) 2020 ATHENA AUTHORS; Ne Luo
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
# pylint: disable=invalid-name
import sys
import codecs

def process_files(decode_log_file, vocab_file):
    """ process decode log, generate label file and results files
    """
    vocab = {}
    with codecs.open(vocab_file, "r", "utf-8") as vocab_file:
        for line in vocab_file:
            phone, num = line.strip().split()
            vocab[int(num)] = phone

    decode_result = open(decode_log_file + ".result", "w", encoding="utf8")
    label_result = open(decode_log_file + ".label", "w", encoding="utf8")

    with open(decode_log_file, "r") as fin:
        to_continue = False
        total_line = ""
        for line in fin.readlines():
            if "predictions" in line:
                total_line = line.strip() + " "
                to_continue = True
            elif to_continue:
                total_line += line.strip() + " "
            if "avg_acc" in total_line and "Message" not in total_line:
                predictions = [int(item) for item in
                    total_line.split("[[")[1].split("]]")[0].split()][:-1]
                labels = [int(item) for item in
                    total_line.split("[[")[2].split("]]")[0].split()]

                decode_result.write(" ".join(
                    " ".join(vocab[item] for item in predictions).split()) + "\n")
                label_result.write(" ".join(
                    " ".join(vocab[item] for item in labels).split()) + "\n")

                decode_result.flush()
                label_result.flush()
                to_continue = False
                total_line = ""
    decode_result.close()
    label_result.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_decode_result.py inference.log vocab")
        sys.exit()

    _, decode_log, vocab = sys.argv
    process_files(decode_log, vocab)
