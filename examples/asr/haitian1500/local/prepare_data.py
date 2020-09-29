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
""" haitian1500 dataset
"""
import os
import re
import sys
import tarfile
import tempfile
import inflect
import codecs
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from absl import logging

import tensorflow as tf
from athena import get_wave_file_length

GFILE = tf.compat.v1.gfile

def download_and_extract(directory, url):
    """Download and extract the given dataset.
    Args:
        directory: the directory where to extract the tarball.
        url: the url to download the data file.
    """
    _, tar_filepath = tempfile.mkstemp(suffix=".tar")  # get rar_path

    try:
        logging.info("Downloading %s to %s" % (url, tar_filepath))

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                "\r>> Downloading {} {:.1f}%".format(
                    tar_filepath, 100.0 * count * block_size / total_size
                )
            )
            sys.stdout.flush()

        urllib.request.urlretrieve(url, tar_filepath, _progress)  # show the progress of download
        statinfo = os.stat(tar_filepath)  # run a stat
        logging.info(
            "Successfully downloaded %s, size(bytes): %d" % (url, statinfo.st_size)  # size -->bytes
        )
        with tarfile.open(tar_filepath, "r") as tar:
            tar.extractall(directory)
    finally:
        GFILE.Remove(tar_filepath)

def get_total_transcript(sub_dir, output_dir, file_name): #合并adult/young所有的txt
    
    logging.info("Processing get_total_transcriptt for {}".format("all_files"))
    
    WAVE_dir = os.path.join(sub_dir, "WAVE")
    SCRIPT_dir= os.path.join(sub_dir, "SCRIPT")
    
    script_lists = os.listdir(SCRIPT_dir)
    #script_lists.sort(key=lambda x:int(x[:-4]))
    
    transcript_path = os.path.join(output_dir, file_name + 'transcript.txt')#adult_transcript.txt
    print(len(script_lists))
    f = open(transcript_path, 'w')
    files = []
    for script_list in script_lists:
        trans_file = os.path.join(SCRIPT_dir,script_list)    
        for line in open(trans_file,encoding="gb18030", errors='ignore'):
            files.append(line)
                
    f.writelines(files)
    f.close()  

def convert_audio_and_split_transcript(sub_dir, output_dir, file_name):
    """Convert rar to WAV and split the transcript.
      Args:
    dataset_dir  : the directory which holds the input dataset.
            -----> "examples/asr/haitian1500/data/select_100"
    total_csv_path : the resulting output csv file.

    """
    logging.info("Processing convert_audio_and_split_transcript {}".format("all_files"))
    
    WAVE_dir = os.path.join(sub_dir, "WAVE")
    SCRIPT_dir= os.path.join(sub_dir, "SCRIPT")

    transcript_path = os.path.join(output_dir, file_name + 'transcript.txt')
    csv_path = os.path.join(output_dir, file_name + 'total.csv')#output_dir/adult_total.csv

    #构建wav_id-wav_file的字典
    key = []
    value = []
    speaker_lists = os.listdir(WAVE_dir) #SPEAKERxxx
    for speaker_list in speaker_lists: 
        speaker = speaker_list[7:]     #get speaker-4600
        wave_path = os.path.join(WAVE_dir, speaker_list)
        wave_dir = wave_path + "/SESSION0"
        wave_lists = os.listdir(wave_dir) 
        for wave_list in wave_lists: #get wave
            transcript_id = wave_list[:-4] #key-序号
            key.append(transcript_id)
            wav_file = os.path.join(wave_dir, wave_list) #value-文件路径
            value.append(wav_file)

    dic = dict(zip(key,value))
    
    #read the transcript.txt 
    files = []
    is_sentid_line = True
    for line in open(transcript_path, encoding="UTF-8-sig", errors='ignore'): #序号 文本 声韵母-类似标贝
        if is_sentid_line:
            transcript_id = line.split('\t')[0]
            #print(transcript_id)
            transcript = line.split('\t')[1].strip()
            #print(transcript)
            wav_file = dic[str(transcript_id)]
            speaker = 'S' + transcript_id[1:5]
            wav_length = get_wave_file_length(wav_file)
            files.append((os.path.abspath(wav_file), wav_length, speaker, transcript))
        is_sentid_line = not is_sentid_line

    df = pandas.DataFrame(
            data=files, columns=["wav_filename", "wav_length_ms", "speaker", "transcript"])
    df.to_csv(csv_path, index=False)
    logging.info("Successfully generated total_csv file {}".format(csv_path)) ##output_dir/adult_total.csv    

def split_train_dev_test(output_dir, file_name):
    # get total_csv
    total_csv_path = os.path.join(output_dir, file_name + 'total.csv')#output_dir/adult_total.csv
    data = pandas.read_csv(total_csv_path)
    x, y = data.iloc[:, :3], data.iloc[:, 3:]
    # split train/dev/test---8:1:1
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.2, random_state=0)
    x_test, x_dev, y_test, y_dev = train_test_split(x_rest, y_rest, test_size=0.5, random_state=0)
    # add transcript
    x_train.insert(2, 'transcript', y_train) #transcript 插入在speakr之前
    x_test.insert(2, 'transcript', y_test)
    x_dev.insert(2, 'transcript', y_dev)
    # get csv_path
    train_csv_path = os.path.join(output_dir, file_name + 'train.csv')
    dev_csv_path = os.path.join(output_dir, file_name + 'dev.csv')
    test_csv_path = os.path.join(output_dir, file_name + 'test.csv')
    # generate csv
    x_train.to_csv(train_csv_path, index=False, sep="\t")
    logging.info("Successfully generated train_csv file {}".format(train_csv_path))
    x_dev.to_csv(dev_csv_path, index=False, sep="\t")
    logging.info("Successfully generated dev_csv file {}".format(dev_csv_path))
    x_test.to_csv(test_csv_path, index=False, sep="\t")
    logging.info("Successfully generated test_csv file {}".format(test_csv_path))

def all_fun(out_dir, in_dir, file_name):
    '''
    out_dir- "examples/asr/haitian1500/data/"
    in_dir- "examples/asr/haitian1500/data/select_100/09_17_ + child/young/adult"
    '''  
    get_total_transcript(in_dir, out_dir, file_name)               #get adult_transcript.txt
    convert_audio_and_split_transcript(in_dir, out_dir, file_name) #get adult_total.csv
    split_train_dev_test(out_dir, file_name)               #get adult_train/dev/test/csv

def processor(dircetory):
    """ download and process """
    
    
    haitian = os.path.join(dircetory, "haitian1500.rar")
    if os.path.exists(haitian):
        logging.info("{} already exist".format(haitian))
    else:
        download_and_extract(dircetory, URL)
    
    # dircetory- "examples/asr/open_source1/data/select_100"
    logging.info("Processing the haitian tcsv in {}".format(dircetory))  
    
    child_file_name = 'child_'
    child_dir = os.path.join(dircetory, "select_100/09_17_child/") 

    young_file_name = 'young_'      
    young_dir = os.path.join(dircetory, "select_100/09_17_young/")

    adult_file_name = 'adult_'
    adult_dir = os.path.join(dircetory, "select_100/09_17_adult/") 

    all_fun(dircetory, child_dir, child_file_name)
    all_fun(dircetory, young_dir, young_file_name)
    all_fun(dircetory, adult_dir, adult_file_name)

    logging.info("Finished processing haitian csv ")


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    INPUT_DIR = sys.argv[1] 
    processor(INPUT_DIR)
