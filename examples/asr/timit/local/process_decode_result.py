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
# processing decode result for timit corpus, mapping phones to target amount
import sys
import codecs

def process_files(decode_log_file, phone_map_file, phone_vocab_file, phone_map_amount):
    """ process decode log, generate label file and results files (original / mapped)
    """
    phone_vocab = {}
    with codecs.open(phone_vocab_file, "r", "utf-8") as vocab_file:
        for line in vocab_file:
            phone, num = line.strip().split()
            phone_vocab[int(num)] = phone

    phone_map = {}
    with codecs.open(phone_map_file, "r", "utf-8") as phones:
        for line in phones:
            # handle phones that are mapped to empty
            if len(line.strip().split("\t")) == 1:
                phone = line.strip()
                phone_48 = ""
                phone_39 = ""
            else:
                phone, phone_48, phone_39 = line.strip().split("\t", 2)
            if phone_map_amount == "48":
                phone_map[phone_48] = phone_39
            elif phone_map_amount == "39":
                phone_map[phone_39] = phone_39
            else:
                phone_map[phone] = phone_39

    decode_result = open(decode_log_file + ".result", "w", encoding="utf8")
    decode_map_result = open(decode_log_file + ".result.map", "w", encoding="utf8")
    label_result = open(decode_log_file + ".label", "w", encoding="utf8")
    label_map_result = open(decode_log_file + ".label.map", "w", encoding="utf8")

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
                    " ".join(phone_vocab[item] for item in predictions).split()) + "\n")
                decode_map_result.write(" ".join(
                    " ".join(phone_map[phone_vocab[item]] for item in predictions).split()) + "\n")

                label_result.write(" ".join(
                    " ".join(phone_vocab[item] for item in labels).split()) + "\n")
                label_map_result.write(" ".join(
                    " ".join(phone_map[phone_vocab[item]] for item in labels).split()) + "\n")

                decode_result.flush()
                label_result.flush()
                decode_map_result.flush()
                label_map_result.flush()
                to_continue = False
                total_line = ""
    decode_result.close()
    label_result.close()
    decode_map_result.close()
    label_map_result.close()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python process_decode_result.py inference.log phones.60-48-39.map" +
              "phone.vocab phone_map_amount")
        sys.exit()

    _, decode_log, phones_map, vocab, map_amount = sys.argv
    process_files(decode_log, phones_map, vocab, map_amount)
