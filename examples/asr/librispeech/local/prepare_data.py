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
""" librspeech """

import codecs
import fnmatch
import os
import sys
import tarfile
import tempfile
import unicodedata

import pandas
import tensorflow as tf
from absl import logging

from six.moves import urllib
from sox import Transformer
from athena import get_wave_file_length

gfile = tf.compat.v1.gfile

SUBSETS = {
    "train-clean-100": "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "http://www.openslr.org/resources/12/train-other-500.tar.gz",
    "dev-clean": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "http://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "http://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "http://www.openslr.org/resources/12/test-other.tar.gz",
}


def download_and_extract(directory, url):
    """Download and extract the given split of dataset.

    Args:
        directory: the directory where to extract the tarball.
        url: the url to download the data file.
    """
    if not gfile.Exists(directory):
        gfile.MakeDirs(directory)

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
        gfile.Remove(tar_filepath)


def convert_audio_and_split_transcript(input_dir,
                                       source_name,
                                       target_name,
                                       output_dir,
                                       output_file):
    """Convert FLAC to WAV and split the transcript.
    Args:
        input_dir: the directory which holds the input dataset.
        source_name: the name of the specified dataset. e.g. test-clean
        target_name: the directory name for the newly generated audio files.
                 e.g. test-clean-wav
        output_dir: the directory to place the newly generated csv files.
        output_file: the name of the newly generated csv file. e.g. test-clean.csv
    """

    logging.info("Processing audio and transcript for %s" % source_name)
    source_dir = os.path.join(input_dir, source_name)
    target_dir = os.path.join(input_dir, target_name)

    if not gfile.Exists(target_dir):
        gfile.MakeDirs(target_dir)

    files = []
    tfm = Transformer()
    # Convert all FLAC file into WAV format. At the same time, generate the csv
    for root, _, filenames in gfile.Walk(source_dir):
        for filename in fnmatch.filter(filenames, "*.trans.txt"):
            trans_file = os.path.join(root, filename)
            with codecs.open(trans_file, "r", "utf-8") as fin:
                for line in fin:
                    seqid, transcript = line.split(" ", 1)
                    # We do a encode-decode transformation here because the output type
                    # of encode is a bytes object, we need convert it to string.
                    transcript = (
                        unicodedata.normalize("NFKD", transcript)
                        .encode("ascii", "ignore")
                        .decode("ascii", "ignore")
                        .strip()
                        .lower()
                    )

                    # Convert FLAC to WAV.
                    flac_file = os.path.join(root, seqid + ".flac")
                    wav_file = os.path.join(target_dir, seqid + ".wav")
                    if not gfile.Exists(wav_file):
                        tfm.build(flac_file, wav_file)
                    # wav_filesize = os.path.getsize(wav_file)
                    wav_length = get_wave_file_length(wav_file)

                    files.append((os.path.abspath(wav_file), wav_length, transcript))
    # Write to CSV file which contains three columns:
    # "wav_filename", "wav_length_ms", "transcript".
    csv_file_path = os.path.join(output_dir, output_file)
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript"]
    )
    df.to_csv(csv_file_path, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(csv_file_path))


def processor(dircetory, subset, force_process):
    """ download and process """
    librispeech_urls = SUBSETS
    if subset not in librispeech_urls:
        raise ValueError(subset, "is not in Librispeech")

    subset_csv = os.path.join(dircetory, subset + ".csv")
    if not force_process and os.path.exists(subset_csv):
        return subset_csv

    dataset_dir = os.path.join(dircetory, subset)
    logging.info("Downloading and process the librispeech in", dataset_dir)
    logging.info("Preparing dataset %s", subset)
    download_and_extract(dataset_dir, librispeech_urls[subset])
    convert_audio_and_split_transcript(
        dataset_dir + "/LibriSpeech",
        subset,
        subset + "-wav",
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
