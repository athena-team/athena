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
# pylint: disable=invalid-name
""" hkust dataset """

import os
import re
import sys
import codecs
from tempfile import TemporaryDirectory
import fnmatch
import pandas
from absl import logging

import tensorflow as tf
from sox import Transformer
from athena import get_wave_file_length
from athena import TextFeaturizer

SUBSETS = ["train", "dev"]


def normalize_hkust_trans(trans):
    """ normalize HKUST transcripts
  delete unuseful symbols and keep space between English words
  Args:
      trans: original transcripts
  """
    norm_trans = re.sub(r"<.*?>|{.*?}", r"", trans)
    tmp_trans = ""
    for item in norm_trans.strip().split():
        if re.search("[a-zA-Z]", item):
            item = " " + item + " "
        tmp_trans += item
    norm_trans = re.sub(r"\s{2,}", r" ", tmp_trans)
    norm_trans = norm_trans.strip()
    return norm_trans


def convert_audio_and_split_transcript(directory, subset, out_csv_file):
    """Convert SPH to WAV and split the transcript.

  Args:
    directory: the directory which holds the input dataset.
    subset: the name of the specified dataset. e.g. dev
    out_csv_file: the resulting output csv file
  """
    gfile = tf.compat.v1.gfile
    sph2pip = os.path.join(os.path.dirname(__file__), "../../../../athena/tools/sph2pipe")
    text_featurizer = TextFeaturizer()

    logging.info("Processing audio and transcript for %s" % subset)
    audio_dir = os.path.join(directory, "LDC2005S15/")
    trans_dir = os.path.join(directory, "LDC2005T32/")

    output_wav_dir = os.path.join(directory, subset + "/wav")
    if not gfile.Exists(output_wav_dir):
        gfile.MakeDirs(output_wav_dir)

    files = []
    char_dict = {}

    sph_files_dict = {}
    for root, _, filenames in gfile.Walk(audio_dir):
        for filename in fnmatch.filter(filenames, "*.sph"):
            if subset in root:
                sph_key = os.path.splitext(filename)[0]
                sph_file = os.path.join(root, filename)
                sph_files_dict[sph_key] = sph_file

    # Convert all SPH file into WAV format.
    # Generate the JSON file and char dict file.
    with TemporaryDirectory(dir="/tmp-data/tmp/") as output_tmp_wav_dir:
        for root, _, filenames in gfile.Walk(trans_dir):
            if not re.match('.*/' + subset + '.*', root):
                continue
            for filename in fnmatch.filter(filenames, "*.txt"):
                trans_file = os.path.join(root, filename)
                sph_key = ""
                speaker_A = ""
                speaker_B = ""
                with codecs.open(trans_file, "r", "gb18030") as fin:
                    for line in fin:
                        line = line.strip()
                        if len(line.split(" ")) <= 1:
                            continue
                        if len(line.split(" ")) == 2:
                            sph_key = line.split(" ")[1]
                            speaker_A = sph_key.split("_")[2]
                            speaker_B = sph_key.split("_")[3]
                            continue

                        time_start, time_end, speaker, transcript = line.split(" ", 3)
                        time_start = float(time_start)
                        time_end = float(time_end)
                        # too short, skip the wave file
                        if time_end - time_start <= 0.1:
                            continue

                        speaker = speaker[0]  # remove ':' in 'A:'
                        if speaker == "A":
                            channel = 1
                            speaker_id = speaker_A
                        else:
                            channel = 2
                            speaker_id = speaker_B

                        # Convert SPH to split WAV.
                        sph_file = sph_files_dict[sph_key]
                        wav_file = os.path.join(
                            output_tmp_wav_dir, sph_key + "." + speaker[0] + ".wav"
                        )
                        if not gfile.Exists(sph_file):
                            raise ValueError(
                                "the sph file {} is not exists".format(sph_file)
                            )
                        if not gfile.Exists(wav_file):
                            sph2pipe_cmd = (
                                sph2pip
                                + " -f wav -c {} -p ".format(str(channel))
                                + sph_file
                                + " "
                                + wav_file
                            )
                            os.system(sph2pipe_cmd)

                        sub_wav_filename = "{0}-{1}-{2:06d}-{3:06d}".format(
                            sph_key, speaker, int(time_start * 100), int(time_end * 100)
                        )
                        sub_wav_file = os.path.join(
                            output_wav_dir, sub_wav_filename + ".wav"
                        )
                        if not gfile.Exists(sub_wav_file):
                            tfm = Transformer()
                            tfm.trim(time_start, time_end)
                            tfm.build(wav_file, sub_wav_file)

                        wav_length = get_wave_file_length(sub_wav_file)

                        transcript = normalize_hkust_trans(transcript)
                        transcript = text_featurizer.delete_punct(transcript)

                        if len(transcript) > 0:
                            for char in transcript:
                                if char in char_dict:
                                    char_dict[char] += 1
                                else:
                                    char_dict[char] = 0
                            files.append(
                                (
                                    os.path.abspath(sub_wav_file),
                                    wav_length,
                                    transcript,
                                    speaker_id,
                                )
                            )

    # Write to CSV file which contains three columns:
    # "wav_filename", "wav_length_ms", "labels".
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript", "speaker"]
    )
    df.to_csv(out_csv_file, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(out_csv_file))


def processor(dircetory, subset, force_process):
    """ download and process """
    if subset not in SUBSETS:
        raise ValueError(subset, "is not in HKUST")

    subset_csv = os.path.join(dircetory, subset + ".csv")
    if not force_process and os.path.exists(subset_csv):
        return subset_csv
    logging.info("Processing the HKUST subset {} in {}".format(subset, dircetory))
    convert_audio_and_split_transcript(dircetory, subset, subset_csv)
    logging.info("Finished processing HKUST subset {}".format(subset))
    return subset_csv


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 2:
        print('Usage: python {} data_dir (data_dir should contain audio and text '
              'directory: LDC2005S15 and LDC2005T32)'.format(sys.argv[0]))
        exit(1)
    DIR = sys.argv[1]
    for SUBSET in SUBSETS:
        processor(DIR, SUBSET, True)
