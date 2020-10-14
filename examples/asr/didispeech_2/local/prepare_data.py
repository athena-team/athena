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
""" didispeech-2 dataset
"""
import os
import re
import sys
import tarfile
import tempfile
import inflect
import codecs
import pandas
from absl import logging
from sklearn.model_selection import train_test_split
from absl import logging
from unidecode import unidecode

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
        
def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
        return True
    else:
        return False
def is_Qalphabet(uchar):
    """判断一个unicode是否是全角英文字母"""
    if (uchar >= u'\uff21' and uchar <= u'\uff3a') or (uchar >= u'\uff41' and uchar <= u'\uff5a'):
        return True
    else:
        return False
def Q2B(uchar):
    """单个字符 全角转半角"""
    new_char = uchar.lower()#转小写
    inside_code = ord(new_char)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
        return new_char
    return chr(inside_code)
def stringpartQ2B(ustring):
    """把字符串中大写字母全角转小写半角"""
    return "".join([Q2B(uchar) if is_Qalphabet(uchar) or is_alphabet(uchar) else uchar for uchar in ustring])

def convert_audio_and_split_transcript(dataset_dir):
    """Convert rar to WAV and split the transcript.
      Args:
    dataset_dir  : the directory which holds the input dataset.
            -----> "examples/asr/didispeech-2/data/
    total_csv_path : the resulting output csv file.

    """
    logging.info("Processing convert_audio_and_split_transcript {}".format("all_files"))
    
    WAVE_dir = os.path.join(dataset_dir, "WAVE")
    SCRIPT_dir= os.path.join(dataset_dir, "SCRIPT")
    csv_path = os.path.join(dataset_dir, 'total.csv')#output_dir/adult_total.csv

    #构建wav_id-wav_file的字典
    key = []
    value = []
    wav_lists = os.listdir(WAVE_dir) #SPEAKERxxx
    for wav_list in wav_lists: 
        transcript_id = wav_list[:-4]#00004944-00000065
        wav_file = os.path.join(WAVE_dir, wav_list)#00004944-00000065.wav
        key.append(transcript_id)
        value.append(wav_file)

    dic = dict(zip(key,value))
    
    files = []
    is_sentid_line = True
    for line in open(transcript_path, encoding="UTF-8-sig", errors='ignore'): #序号 文本 声韵母-类似标贝
        if is_sentid_line:
            transcript_id = line.split('\t')[0]
            speaker = transcript_id.split('-')[0]
            transcript = line.split('\t')[1].strip()
            transcript = stringpartQ2B(transcript)#大写全角转小写半角
            #去除符号
            punc = '，。！？【】（）《》“‘：；［］｛｝&，．？（）＼％－＋￣~＄#＠=＿、／\\%-+~~$#@=_//,.!?' 
            transcript = re.sub(r"[%s]+" %punc, "", transcript)
            #print(transcript)           
            wav_file = dic[str(transcript_id)]
            wav_length = get_wave_file_length(wav_file)
            files.append((os.path.abspath(wav_file), wav_length, speaker, transcript))
        is_sentid_line = not is_sentid_line

    df = pandas.DataFrame(
            data=files, columns=["wav_filename", "wav_length_ms", "speaker", "transcript"])
    df.to_csv(csv_path, index=False)
    logging.info("Successfully generated total_csv file {}".format(csv_path)) ##output_dir/adult_total.csv    

def split_train_dev_test(output_dir):
    # get total_csv
    total_csv_path = os.path.join(output_dir, 'total.csv')#output_dir/adult_total.csv
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

def processor(dircetory):
    """ download and process """   
    haitian = os.path.join(dircetory, "didispeech-2.rar")
    if os.path.exists(haitian):
        logging.info("{} already exist".format(haitian))
    else:
        download_and_extract(dircetory, URL)
        
    convert_audio_and_split_transcript(dircetory)
    split_train_dev_test(dircetory)
    logging.info("Finished processing didispeech csv ")
    
if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    INPUT_DIR = sys.argv[1] 
    processor(INPUT_DIR)
