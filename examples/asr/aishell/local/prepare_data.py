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

""" aishell dataset """

import os
import sys
import codecs
import pandas
from absl import logging

import tensorflow as tf
from athena import get_wave_file_length

SUBSETS = ["train", "dev", "test"]


def convert_audio_and_split_transcript(dataset_dir, subset, out_csv_file):
    """Convert tar.gz to WAV and split the transcript.

  Args:
    dataset_dir  : the directory which holds the input dataset.
    subset       : the name of the specified dataset. e.g. dev.
    out_csv_file : the resulting output csv file.
  """

    gfile = tf.compat.v1.gfile
    logging.info("Processing audio and transcript for {}".format(subset))
    audio_dir = os.path.join(dataset_dir, "wav/")
    trans_dir = os.path.join(dataset_dir, "transcript/")

    files = []
    char_dict = {}
    if not gfile.Exists(os.path.join(audio_dir, subset)): # not unzip wav yet
        for filename in os.listdir(audio_dir):
            os.system("tar -zxvf " + audio_dir + filename + " -C " + audio_dir)

    with codecs.open(os.path.join(trans_dir, "aishell_transcript_v0.8.txt"), "r", encoding="utf-8") as f:
        for line in f:
            items = line.strip().split(" ")
            wav_filename = items[0]
            labels = ""
            for item in items[1:]:
                labels += item
                if item in char_dict:
                    char_dict[item] += 1
                else:
                    char_dict[item] = 0
            files.append((wav_filename + ".wav", labels))
    files_size_dict = {}
    output_wav_dir = os.path.join(audio_dir, subset)

    for root, subdirs, _ in gfile.Walk(output_wav_dir):
        for subdir in subdirs:
            for filename in os.listdir(os.path.join(root, subdir)):
                files_size_dict[filename] = (
                    get_wave_file_length(os.path.join(root, subdir, filename)),
                    subdir,
                )

    content = []
    for wav_filename, trans in files:
        if wav_filename in files_size_dict: # wav which has trans is valid
            filesize, subdir = files_size_dict[wav_filename]
            abspath = os.path.join(output_wav_dir, subdir, wav_filename)
            content.append((abspath, filesize, trans, subdir))
    files = content

    # Write to CSV file which contains three columns:
    # "wav_filename", "wav_length_ms", "transcript", "speakers".
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript", "speaker"]
    )
    df.to_csv(out_csv_file, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(out_csv_file))

def processor(dataset_dir, subset, force_process, output_dir):
    """ download and process """
    if subset not in SUBSETS:
        raise ValueError(subset, "is not in AISHELL")
    if force_process:
        logging.info("force process is set to be true")

    subset_csv = os.path.join(output_dir, subset + ".csv")
    if not force_process and os.path.exists(subset_csv):
        logging.info("{} already exist".format(subset_csv))
        return subset_csv
    logging.info("Processing the AISHELL subset {} in {}".format(subset, dataset_dir))
    convert_audio_and_split_transcript(dataset_dir, subset, subset_csv)
    logging.info("Finished processing AISHELL subset {}".format(subset))
    return subset_csv

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        logging.info('Usage: python {} dataset_dir output_dir\n'
              '    dataset_dir : directory contains AISHELL dataset\n'
              '    output_dir  : Athena working directory'.format(sys.argv[0]))
        exit(1)
    DATASET_DIR = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    for SUBSET in SUBSETS:
        processor(DATASET_DIR, SUBSET, True, OUTPUT_DIR)
