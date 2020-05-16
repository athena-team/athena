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
""" data baker dataset
Speech corpus is a female who pronounces standard Mandarin.
It includes about 12 hours and is composed of 10000 sentences.
detailed information can be seen on https://www.data-baker.com/open_source.html
"""

import os
import sys
import codecs
from string import digits
import tempfile
import re
import rarfile
import pandas
from six.moves import urllib
from absl import logging
from sklearn.model_selection import train_test_split

import tensorflow as tf
from athena import get_wave_file_length

# SUBSETS = ["train", "dev", "test"]
GFILE = tf.compat.v1.gfile

URL = "https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar"


def normalize_biaobei_trans(trans):
  ''' normalize HKUST transcripts
  delete unuseful symbols and keep space between English words
  Args:
      trans: original transcripts
  '''
  norm_trans = re.sub(r'[，、”“：？！（）……]', r'', trans)
  tmp_trans = ''
  for item in norm_trans.strip().split():
    if re.search('[a-zA-Z]', item):
      item = ' ' + item + ' '
    tmp_trans += item
  norm_trans = re.sub(r'\s{2,}', r' ', tmp_trans)
  norm_trans = norm_trans.strip()
  return norm_trans


def download_and_extract(directory, url):
    """Download and extract the given dataset.
    Args:
        directory: the directory where to extract the tarball.
        url: the url to download the data file.
    """
    _, rar_filepath = tempfile.mkstemp(suffix=".rar")  # get rar_path

    try:
        logging.info("Downloading %s to %s" % (url, rar_filepath))

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                "\r>> Downloading {} {:.1f}%".format(
                    rar_filepath, 100.0 * count * block_size / total_size
                )
            )
            sys.stdout.flush()

        urllib.request.urlretrieve(url, rar_filepath, _progress)  # show the progress of download
        statinfo = os.stat(rar_filepath)  # run a stat
        logging.info(
            "Successfully downloaded %s, size(bytes): %d" % (url, statinfo.st_size)  # size -->bytes
        )
        rf = rarfile.RarFile(rar_filepath)  # need to install unrar!!!!
        rf.extractall(directory)
    finally:
        GFILE.Remove(rar_filepath)


def convert_audio_and_split_transcript(dataset_dir, total_csv_path):
    """Convert rar to WAV and split the transcript.
      Args:
    dataset_dir  : the directory which holds the input dataset.
            -----> /nfs/project/datasets/data_baker_tts
    total_csv_path : the resulting output csv file.

    BZNSYP dir Tree structure:
    BZNSYP
        -ProsodyLabeling
            -000001-010000.txt
        -Wave
            -000001.wav
            -000002.wav
            ...
        -PhoneLabeling
            -000001.interval
            -000002.interval
            ...
    """
    logging.info("ProcessingA audio and transcript for {}".format("all_files"))
    audio_dir = os.path.join(dataset_dir, "BZNSYP/Wave/")
    prosodyLabel_dir = os.path.join(dataset_dir, "BZNSYP/ProsodyLabeling/")

    files = []
    # ProsodyLabel ---word
    with codecs.open(os.path.join(prosodyLabel_dir, "000001-010000.txt"),
                     "r",
                     encoding="utf-8") as f:
        for line in f:
            if line[0:1] == '0':
                # 000001	卡尔普#2陪外孙#1玩滑梯#4。
                line = line[:-3]  # remove "#4。"
                res = line.replace('#', '')
                wav_filename = res[0:6]  # 000001
                # get wav_length
                wav_file = os.path.join(audio_dir, wav_filename + '.wav')
                wav_length = get_wave_file_length(wav_file)
                # remove digit
                remove_digits = str.maketrans('', '', digits)
                item = res.translate(remove_digits).strip()  # 卡尔普陪外孙玩滑梯
                transcript = ""
                transcript += item
                transcript = normalize_hkust_trans(transcript)
                # [('000001.wav', ' 卡尔普陪外孙玩滑梯')...]
                files.append((os.path.abspath(wav_file), wav_length, transcript))

    # Write to CSV file which contains three columns:
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript"]
    )
    df.to_csv(total_csv_path, index=False)
    logging.info("Successfully generated csv file {}".format(total_csv_path))


def split_train_dev_test(total_csv, output_dir):
    # get total_csv
    data = pandas.read_csv(total_csv)
    x, y = data.iloc[:, :2], data.iloc[:, 2:]
    # split train/dev/test---8:1:1
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.2, random_state=0)
    x_test, x_dev, y_test, y_dev = train_test_split(x_rest, y_rest, test_size=0.5, random_state=0)
    # add ProsodyLabel
    x_train.insert(2, 'transcript', y_train)
    x_test.insert(2, 'transcript', y_test)
    x_dev.insert(2, 'transcript', y_dev)
    # get csv_path
    train_csv_path = os.path.join(output_dir, 'train.csv')
    dev_csv_path = os.path.join(output_dir, 'dev.csv')
    test_csv_path = os.path.join(output_dir, 'test.csv')
    # generate csv
    x_train.to_csv(train_csv_path, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(train_csv_path))
    x_dev.to_csv(dev_csv_path, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(dev_csv_path))
    x_test.to_csv(test_csv_path, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(test_csv_path))


def processor(dircetory):
    """ download and process """
    # already downloaded data_baker
    data_baker = os.path.join(dircetory, "BZNSYP.rar")
    if os.path.exists(data_baker):
        logging.info("{} already exist".format(data_baker))
    else:
        download_and_extract(dircetory, URL)

    # get total_csv
    logging.info("Processing the BiaoBei total.csv in {}".format(dircetory))
    total_csv_path = os.path.join(dircetory, "total.csv")
    convert_audio_and_split_transcript(dircetory, total_csv_path)
    # split 8:1:1
    split_train_dev_test(total_csv_path, dircetory)
    logging.info("Finished processing BiaoBei csv ")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    DIR = sys.argv[1]
    processor(DIR)
