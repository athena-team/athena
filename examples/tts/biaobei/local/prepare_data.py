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
""" biaobei dataset """


import os
import sys
import codecs
import pandas
import string
from string import digits
import rarfile
import tempfile
from six.moves import urllib
from absl import logging
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from athena import get_wave_file_length

#SUBSETS = ["train", "dev", "test"]
gfile = tf.compat.v1.gfile

biaobei_tts_urls = "https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar"

def download_and_extract(directory, url):
    """Download and extract the given dataset.
    Args:
        directory: the directory where to extract the tarball.
        url: the url to download the data file.
    """
    if not gfile.Exists(directory):
        gfile.MakeDirs(directory)

    _, rar_filepath = tempfile.mkstemp(suffix=".rar")#get rar_path

    try:
        logging.info("Downloading %s to %s" % (url, rar_filepath))

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                "\r>> Downloading {} {:.1f}%".format(
                    rar_filepath, 100.0 * count * block_size / total_size
                )
            )
            sys.stdout.flush()

        urllib.request.urlretrieve(url, rar_filepath, _progress)#show the progress of download
        statinfo = os.stat(rar_filepath)#run a stat
        logging.info(
            "Successfully downloaded %s, size(bytes): %d" % (url, statinfo.st_size)# size -->bytes
        )
        rf = rarfile.RarFile(rar_filepath)#need to install unrar!!!!
        rf.extractall(directory)
    finally:
        gfile.Remove(rar_filepath)

def convert_audio_and_split_transcript(dataset_dir, total_csv_path):
    """Convert rar to WAV and split the transcript.
      Args:
    dataset_dir  : the directory which holds the input dataset.
            -----> /nfs/project/datasets/biaobei_tts
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
    gfile = tf.compat.v1.gfile #file_io
    logging.info("ProcessingA audio and transcript for {}".format("all_files"))
    audio_dir = os.path.join(dataset_dir, "BZNSYP/Wave/") 
    ProsodyLabel_dir = os.path.join(dataset_dir, "BZNSYP/ProsodyLabeling/")
    #PhoneLabel_dir = os.path.join(dataset_dir, "PhoneLabeling/")

    files = []
    # ProsodyLabel ---word  
    with codecs.open(os.path.join(ProsodyLabel_dir, "000001-010000.txt"), "r", encoding="utf-8") as f:
        for line in f:
            if line[0:1] == '0': 
                # 000001	卡尔普#2陪外孙#1玩滑梯#4。
                line = line[:-3] # remove "#4。"
                res = line.replace('#', '') 
                wav_filename = res[0:6] # 000001
                #get wav_length
                wav_file = os.path.join(audio_dir, wav_filename + '.wav')
                wav_length = get_wave_file_length(wav_file)
                #remove digit
                remove_digits = str.maketrans('', '', digits)
                item = res.translate(remove_digits).strip() # 卡尔普陪外孙玩滑梯                  
                transcript = ""
                transcript += item
                files.append((os.path.abspath(wav_file), wav_length, transcript)) #[('000001.wav', ' 卡尔普陪外孙玩滑梯')...]

    # Write to CSV file which contains three columns:
    # "wav_filename", "wav_length_ms", "transcript"
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript"]
    )
    df.to_csv(total_csv_path, index=False)
    logging.info("Successfully generated csv file {}".format(total_csv_path))

def split_train_dev_test(total_csv,output_dir):
    #get total_csv
    data = pd.read_csv(total_csv)
    x,y = data.iloc[:,0:2], data.iloc[:,2:] 
    #split train/dev/test---8:1:1
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.2,random_state=0)
    x_test, x_dev, y_test, y_dev = train_test_split(x_rest, y_rest, test_size=0.5,random_state=0)
    #add ProsodyLabel
    x_train.insert(2,'ProsodyLabel', y_train)
    x_test.insert(2,'ProsodyLabel', y_test)
    x_dev.insert(2,'ProsodyLabel', y_dev)
    #get csv_path
    train_csv_path = os.path.join(output_dir, 'train.csv')
    dev_csv_path = os.path.join(output_dir, 'dev.csv')
    test_csv_path = os.path.join(output_dir, 'test.csv')
    #generate csv
    x_train.to_csv(train_csv_path, index=False,sep="\t")
    logging.info("Successfully generated csv file {}".format(train_csv_path))
    x_dev.to_csv(dev_csv_path, index=False,sep="\t")
    logging.info("Successfully generated csv file {}".format(dev_csv_path))
    x_test.to_csv(test_csv_path, index=False,sep="\t")
    logging.info("Successfully generated csv file {}".format(test_csv_path))

def processor(dataset_dir, output_dir):
    """ download and process """  
    #already downloaded biaobei 
    biaobei = os.path.join(dataset_dir, "BZNSYP.rar")
    if os.path.exists(biaobei):
        logging.info("{} already exist".format(biaobei))
        
        #rf = rarfile.RarFile(biaobei)#need to install unrar!!!!
        #rf.extractall(dataset_dir)
    else:
        download_and_extract(dataset_dir, biaobei_tts_urls)

    #get total_csv
    logging.info("Processing the BiaoBei total.csv in {}".format(dataset_dir))
    total_csv_path = os.path.join(output_dir, "total.csv")
    convert_audio_and_split_transcript(dataset_dir, total_csv_path)
    #split 8:1:1
    split_train_dev_test(total_csv_path, output_dir)
    logging.info("Finished processing BiaoBei csv ")

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        print('Usage: python {} dataset_dir output_dir\n'
              '    dataset_dir : directory contains BiaoBei dataset\n'
              '    output_dir  : Athena working directory'.format(sys.argv[0]))
        exit(1)
    DATASET_DIR = sys.argv[1]#/nfs/project/datasets/biaobei_tts
    OUTPUT_DIR = sys.argv[2] 
    processor(DATASET_DIR, OUTPUT_DIR)
