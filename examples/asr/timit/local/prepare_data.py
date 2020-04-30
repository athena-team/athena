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
# pylint: disable=invalid-name
""" timit """

import codecs
import fnmatch
import os
import sys
import re
import pandas
import tensorflow as tf
from absl import logging
from athena import get_wave_file_length

SUBSETS = ['train', 'test', 'core-test']
CORE_TESTSET_SPEAKER = ['mdab0', 'mwbt0', 'felc0', 'mtas1', 'mwew0', 'fpas0', 'mjmp0', 
                        'mlnt0', 'fpkt0', 'mlll0', 'mtls0', 'fjlm0', 'mbpm0', 'mklt0', 
                        'fnlp0', 'mcmj0', 'mjdh0', 'fmgd0', 'mgrt0', 'mnjm0', 'fdhc0', 
                        'mjln0', 'mpam0', 'fmld0']

def convert_audio_and_split_transcript(dataset_dir,
                                       subset,
                                       output_dir,
                                       trans_type="phn",
                                       phone_map_amount=None):
    """Convert SPH to WAV and split the transcript.
    Args:
        dataset_dir: the directory which holds the input dataset.
        subset: the name of the specified dataset. e.g. train
        output_dir: the directory to place the newly generated csv files.
        trans_type: the type of transcript. for timit: "char" or "phn".
        phone_map_amount: convert phones according to "phones.60-48-39.map",
                          should be None or "48" or "39".
    """

    logging.info("Processing audio and transcript for %s" % subset)
    csv_file_path = os.path.join(output_dir, subset + ".csv")

    core_test = False
    if subset == "core-test":
        core_test = True
        subset = "test"

    subset_dir = os.path.join(dataset_dir, subset)
    output_wav_dir = os.path.join(output_dir, "wav", subset)
    gfile = tf.compat.v1.gfile
    sph2pipe = os.path.join(os.path.dirname(__file__), "../../../../tools/sph2pipe/sph2pipe")
    phone_map_file = "examples/asr/timit/local/phones.60-48-39.map"

    if not os.path.exists(output_wav_dir):
        os.makedirs(output_wav_dir)

    if trans_type == "char":
        fnmatch_pattern = "*.txt"
    elif trans_type == "phn":
        fnmatch_pattern = "*.phn"
    else:
        raise ValueError(fnmatch_pattern, "is not an effective transcript type in TIMIT")

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
                phone_map[phone] = phone_48
            elif phone_map_amount == "39":
                phone_map[phone] = phone_39
            else:
                phone_map[phone] = phone

    files = []
    for root, _, filenames in gfile.Walk(subset_dir):
        for filename in fnmatch.filter(filenames, fnmatch_pattern):
            trans_file = os.path.join(root, filename)
            with codecs.open(trans_file, "r", "utf-8") as fin:
                if trans_type == "char":
                    for line in fin:
                        begin_sample, end_sample, transcript = line.strip().split(" ", 2)
                        transcript = transcript.lower()
                        transcript = re.sub('[.,?!;:"]', '', transcript)
                else:
                    phns = []
                    for line in fin:
                        begin_sample, end_sample, phn = line.strip().split(" ", 2)
                        phns.append(phn)
                    # join twice to remove extra spaces in transcript caused by empty phones
                    transcript = " ".join(" ".join(phone_map[item] for item in phns).split())

            speaker = os.path.basename(root)
            file_name = os.path.splitext(filename)[0]
            utt_key = speaker + "-" + file_name

            # sphere to wav
            sph_file = os.path.join(root, file_name + ".wav")
            wav_file = os.path.join(output_wav_dir, utt_key + ".wav")
            channel = 1
            if not gfile.Exists(wav_file):
                sph2pipe_cmd = (
                    sph2pipe
                    + " -f wav -c {} -p ".format(str(channel))
                    + sph_file
                    + " "
                    + wav_file
                )
                os.system(sph2pipe_cmd)

            wav_length = get_wave_file_length(wav_file)
            if core_test: 
                if speaker in CORE_TESTSET_SPEAKER and not file_name.startswith("sa"):
                    files.append((os.path.abspath(wav_file), wav_length, transcript, speaker))
            else:
                files.append((os.path.abspath(wav_file), wav_length, transcript, speaker))
    # Write to CSV file which contains four columns:
    # "wav_filename", "wav_length_ms", "transcript", "speaker".
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript", "speaker"]
    )
    df.to_csv(csv_file_path, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(csv_file_path))


def processor(dataset_dir, subset, output_dir, force_process):
    """ download and process """
    if subset not in SUBSETS:
        raise ValueError(subset, "is not in TIMIT")

    subset_csv = os.path.join(output_dir, subset + ".csv")
    if not force_process and os.path.exists(subset_csv):
        return subset_csv

    logging.info(
        "Processing the TIMIT subset {} in {}".format(subset, dataset_dir)
    )
    convert_audio_and_split_transcript(
        dataset_dir,
        subset,
        output_dir,
        trans_type="phn",
        phone_map_amount="39"
    )
    logging.info("Finished processing TIMIT subset {}".format(subset))
    return subset_csv


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        print('Usage: python {} dataset_dir output_dir\n'
              '       dataset_dir : directory contains TIMIT original data\n'
              '       output_dir  : Athena working directory'.format(sys.argv[0]))
        sys.exit()
    DATASET_DIR = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    for SUBSET in SUBSETS:
        processor(DATASET_DIR, SUBSET, OUTPUT_DIR, False)
