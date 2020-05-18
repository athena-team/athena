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
""" libritts dataset

LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech
https://arxiv.org/pdf/1904.02882
"""

import codecs
import fnmatch
import os
import sys
import tarfile
import tempfile

import pandas
import tensorflow as tf
from absl import logging

from six.moves import urllib
from athena import get_wave_file_length

GFILE = tf.compat.v1.gfile

SUBSETS = {
    "train-clean-100": "http://www.openslr.org/resources/60/train-clean-100.tar.gz",
    "train-clean-360": "http://www.openslr.org/resources/60/train-clean-360.tar.gz",
    "train-other-500": "http://www.openslr.org/resources/60/train-other-500.tar.gz",
    "dev-clean": "http://www.openslr.org/resources/60/dev-clean.tar.gz",
    "dev-other": "http://www.openslr.org/resources/60/dev-other.tar.gz",
    "test-clean": "http://www.openslr.org/resources/60/test-clean.tar.gz",
    "test-other": "http://www.openslr.org/resources/60/test-other.tar.gz",
}

def download_and_extract(directory, url):
    """Download and extract the given split of dataset.

    Args:
        directory: the directory where to extract the tarball.
        url: the url to download the data file.
    """
    if not GFILE.Exists(directory):
        GFILE.MakeDirs(directory)

    _, tar_filepath = tempfile.mkstemp(suffix=".tar.gz")

    try:
        logging.info("Downloading %s to %s" % (url, tar_filepath))

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                "\r>> Downloading {} {:.1f}%".format(
                    tar_filepath, 100.0 * count * block_size / total_size
                )
            )
            sys.stdout.flush()

        urllib.request.urlretrieve(url, tar_filepath, _progress)
        statinfo = os.stat(tar_filepath)
        logging.info(
            "Successfully downloaded %s, size(bytes): %d" % (url, statinfo.st_size)
        )
        with tarfile.open(tar_filepath, "r") as tar:
            tar.extractall(directory)
    finally:
        GFILE.Remove(tar_filepath)

def convert_audio_and_split_transcript(input_dir,
                                       source_name,
                                       output_dir,
                                       output_file):

    """
    LibriTTS dir Tree structure:
    root
        -LibriTTS
            - subset
                - book id?
                    - speaker id?
                        - bookid_speakerid_paragraphid_sentenceid.normalized.txt
                        - bookid_speakerid_paragraphid_sentenceid.original.txt
                        - bookid_speakerid_paragraphid_sentenceid.wav
                        ...
    """
    logging.info("Processing audio and transcript for %s" % source_name)
    source_dir = os.path.join(input_dir, source_name)

    files = []
    # generate the csv
    for root, _, filenames in GFILE.Walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.normalized.txt"):
            trans_file = os.path.join(root, filename)
            seqid = filename.split('.')[0]
            speaker = seqid.split('_')[1]
            wav_file = os.path.join(root, seqid + '.wav')
            wav_length = get_wave_file_length(wav_file)

            with codecs.open(trans_file, "r", "utf-8") as fin:
                transcript = fin.readline().strip().lower()
                files.append((os.path.abspath(wav_file), wav_length, transcript, speaker))

    # Write to CSV file which contains four columns:
    # "wav_filename", "wav_length_ms", "transcript", "speaker".
    csv_file_path = os.path.join(output_dir, output_file)
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript", "speaker"]
    )
    df.to_csv(csv_file_path, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(csv_file_path))

def processor(dircetory, subset, force_process):
    """ download and process """
    libritts_urls = SUBSETS
    if subset not in libritts_urls:
        raise ValueError(subset, "is not in LibriTTS")

    subset_csv = os.path.join(dircetory, subset + ".csv")
    if not force_process and os.path.exists(subset_csv):
        return subset_csv

    dataset_dir = dircetory
    logging.info("Downloading and process the libritts in", dataset_dir)
    logging.info("Preparing dataset %s", subset)
    download_and_extract(dataset_dir, libritts_urls[subset])
    convert_audio_and_split_transcript(
        dataset_dir + "/LibriTTS",
        subset,
        dircetory,
        subset + ".csv",
    )
    logging.info("Finished downloading and processing")
    return subset_csv

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    DIR = sys.argv[1]
    for SUBSET in SUBSETS:
        processor(DIR, SUBSET, True)
