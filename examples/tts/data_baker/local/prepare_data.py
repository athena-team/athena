# Copyright (C) ATHENA AUTHORS;Cheng Wen
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
import re
import sys
import codecs
import json
import urllib
import rarfile
import tempfile
import pandas
import subprocess
from absl import logging
from sklearn.model_selection import train_test_split
import tensorflow as tf
from athena import get_wave_file_length
from athena.main import parse_config
from extract_feat import ExtractFeatures
GFILE = tf.compat.v1.gfile
from joblib import Parallel, delayed
URL = "https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar"

# ascii code, used to delete Chinese punctuation
CHN_PUNC_LIST = [183, 215, 8212, 8216, 8217, 8220, 8221, 8230,
                 12289, 12290, 12298, 12299, 12302, 12303, 12304, 12305,
                 65281, 65288, 65289, 65292, 65306, 65307, 65311]
CHN_PUNC_SET = set(CHN_PUNC_LIST)

MANDARIN_INITIAL_LIST = ["b", "ch", "c", "d", "f", "g", "h", "j",
                         "k", "l", "m", "n", "p", "q", "r", "sh", "s", "t", "x", "zh", "z"]

# prosody phone list
CHN_PHONE_PUNC_LIST = ['sp2', 'sp1', 'sil']
# erhua phoneme
CODE_ERX = 0x513F


def _update_insert_pos(old_pos, pylist):
    new_pos = old_pos + 1
    i = new_pos
    while i < len(pylist)-1:
        # if the first letter is upper, then this is the phoneme of English letter
        if pylist[i][0].isupper():
            i += 1
            new_pos += 1
        else:
            break
    return new_pos


def _pinyin_preprocess(line, words):
    if line.find('.') >= 0:
        # remove '.' in English letter phonemes, for example: 'EH1 F . EY1 CH . P IY1'
        py_list = line.replace('/', '').strip().split('.')
        py_str = ''.join(py_list)
        pinyin = py_str.split()
    else:
        pinyin = line.replace('/', '').strip().split()

    # now the content in pinyin like: ['OW1', 'K', 'Y', 'UW1', 'JH', 'EY1', 'shi4', 'yi2', 'ge4']
    insert_pos = _update_insert_pos(-1, pinyin)
    i = 0
    while i < len(words):
        if ord(words[i]) in CHN_PUNC_SET:
            i += 1
            continue
        if words[i] == '#' and (words[i+1] >= '1' and words[i+1] <= '4'):
            if words[i+1] == '1':
                pass
            else:
                if words[i+1] == '2':
                    pinyin.insert(insert_pos, 'sp1')
                if words[i+1] == '3':
                    pinyin.insert(insert_pos, 'sp1')
                elif words[i+1] == '4':
                    pinyin.append('sil')
                    break
                insert_pos = _update_insert_pos(insert_pos, pinyin)
            i += 2
        elif ord(words[i]) == CODE_ERX:
            if pinyin[insert_pos-1].find('er') != 0:  # erhua
                i += 1
            else:
                insert_pos = _update_insert_pos(insert_pos, pinyin)
                i += 1
        # skip non-mandarin characters, including A-Z, a-z, Greece letters, etc.
        elif ord(words[i]) < 0x4E00 or ord(words[i]) > 0x9FA5:
            i += 1
        else:
            insert_pos = _update_insert_pos(insert_pos, pinyin)
            i += 1
    return pinyin


def _pinyin_2_initialfinal(py):
    """
    used to split pinyin into intial and final phonemes
    """
    if py[0] == 'a' or py[0] == 'e' or py[0] == 'E' or py[0] == 'o' or py[:2] == 'ng' or \
            py[:2] == 'hm':
        py_initial = ''
        py_final = py
    elif py[0] == 'y':
        py_initial = ''
        if py[1] == 'u' or py[1] == 'v':
            py_final = list(py[1:])
            py_final[0] = 'v'
            py_final = ''.join(py_final)
        elif py[1] == 'i':
            py_final = py[1:]
        else:
            py_final = list(py)
            py_final[0] = 'i'
            py_final = ''.join(py_final)
    elif py[0] == 'w':
        py_initial = ''
        if py[1] == 'u':
            py_final = py[1:]
        else:
            py_final = list(py)
            py_final[0] = 'u'
            py_final = ''.join(py_final)
    else:
        init_cand = ''
        for init in MANDARIN_INITIAL_LIST:
            init_len = len(init)
            init_cand = py[:init_len]
            if init_cand == init:
                break
        if init_cand == '':
            raise Exception('unexpected')
        py_initial = init_cand
        py_final = py[init_len:]
        if (py_initial in set(['j', 'q', 'x']) and py_final[0] == 'u'):
            py_final = list(py_final)
            py_final[0] = 'v'
            py_final = ''.join(py_final)
    if py_final[-1] == '6':
        py_final = py_final.replace('6', '2')
    return (py_initial, py_final)


def is_all_eng(words):
    # if include mandarin
    for word in words:
        if ord(word) >= 0x4E00 and ord(word) <= 0x9FA5:
            return False
    return True


def pinyin_2_phoneme(pinyin_line, words):
    #chn or chn+eng
    sent_phoneme = ['sil']
    if not is_all_eng(words):
        sent_py = _pinyin_preprocess(pinyin_line, words)
        for py in sent_py:
            if py[0].isupper() or py in CHN_PHONE_PUNC_LIST:
                sent_phoneme.append(py)
            else:
                initial, final = _pinyin_2_initialfinal(py)
                if initial == '':
                    sent_phoneme.append(final)
                else:
                    sent_phoneme.append(initial)
                    sent_phoneme.append(final)
    else:
        wordlist = words.split(' ')
        word_phonelist = pinyin_line.strip().split('/')
        assert(len(word_phonelist) == len(wordlist))
        i = 0
        while i < len(word_phonelist):
            phone = re.split(r'[ .]', word_phonelist[i])
            for p in phone:
                if p:
                    sent_phoneme.append(p)
            if '/' in wordlist[i]:
                sent_phoneme.append('sp1')
            elif '%' in wordlist[i]:
                if i != len(word_phonelist)-1:
                    sent_phoneme.append('sp1')
                else:
                    sent_phoneme.append('sil')
            i += 1
    return ' '.join(sent_phoneme)


def unrar(rar_filepath, directory):
    out_path = os.path.join(directory, 'BZNSYP')
    rfile = rarfile.RarFile(rar_filepath)  # need to install unrar!!!!
    rfile.extractall(out_path)
    for f in rfile.namelist():
        rar_file = os.path.join(out_path, f)
        rf = rarfile.RarFile(rar_file)
        rf.extractall(out_path)


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
                "\r > > Downloading {} {:.1f}%".format(
                    rar_filepath, 100.0 * count * block_size / total_size
                )
            )
            sys.stdout.flush()

        # show the progress of download
        urllib.request.urlretrieve(url, rar_filepath, _progress)
        statinfo = os.stat(rar_filepath)  # run a stat
        logging.info(
            "Successfully downloaded %s, size(bytes): %d" % (
                url, statinfo.st_size)  # size-->bytes
        )
        unrar(rar_filepath, directory)
        logging.info("Successfully extracted data from rar")
    finally:
        GFILE.Remove(rar_filepath)


def process_prosody(dataset_dir):
    """
    Convert phonelabeling to phone dict
    Args:
        dataset_dir
    Returns:
        phone_dict: {sent_id:[phone_str]}   
    """
    trans_path = os.path.join(dataset_dir, "BZNSYP/ProsodyLabeling/")
    is_sentid_line = True
    phone_dict = {}
    with open(trans_path + '000001-010000.txt', encoding='utf-8') as f:
        for line in f:
            if is_sentid_line:
                sent_id = line.split()[0]
                words = line.split('\t')[1].strip()
            else:
                sent_phonemes = pinyin_2_phoneme(line, words)
                phone_dict.update({sent_id: [sent_phonemes]})
            is_sentid_line = not is_sentid_line
    return phone_dict


def process_phonelabeling(dataset_dir):
    """
    Convert phonelabeling to phone_dur dict
    Args:
        dataset_dir
    Returns:
        phone_dur_dict: {utt_id:[phone_str, interval_str]}   
    """
    interval_dir = os.path.join(dataset_dir, "BZNSYP/PhoneLabeling/")
    interval_files = os.listdir(interval_dir)
    phone_dur_dict = {}
    for fn in interval_files:
        interval_file = os.path.join(interval_dir, fn)
        utt_id = os.path.splitext(fn)[0]
        if fn.split('.')[1] != 'interval':
            continue
        with open(interval_file, 'r') as f1:
            lines = f1.readlines()
            interval_list = []
            phones_list = []
            for i in range(12, len(lines), 3):
                dur = str(float(lines[i+1].strip()) - float(lines[i].strip()))
                interval_list.append(dur)
                phone = eval(lines[i+2].strip())
                if phone == 'sp':
                    phone = 'sp1'
                # change tone
                if phone[-1] == '6':
                    phone = phone.replace('6', '3')
                # replace iii  and ii with i
                phone = phone.replace('iii', 'i').replace('ii', 'i')
                phones_list.append(phone)
            phone_str = ' '.join(phones_list)
            interval_str = ' '.join(interval_list)
            phone_dur_dict.update({utt_id: [phone_str,interval_str]})
    return phone_dur_dict


def process_wav(dataset_dir):
    """Downsampling wav and return wav dict
    """
    audio_dir = os.path.join(dataset_dir, "BZNSYP/")
    os.makedirs(os.path.join(audio_dir, "Wave_24"), exist_ok=True)
    wav_files = os.listdir(os.path.join(audio_dir, "Wave"))
    wav_dict = {}

    for wav in wav_files:
        utt_id = os.path.splitext(wav)[0]
        wav48k = os.path.join(audio_dir, 'Wave', wav)
        wav24k = os.path.join(audio_dir, 'Wave_24', wav)
        subprocess.getoutput(
            'cat %s | /usr/bin/sox -t wav - -c 1 -b 16 -t wav - rate 24000  >  %s' % (wav48k, wav24k))
        wav_length = get_wave_file_length(wav24k)
        wav_dict.update({utt_id: [wav24k, wav_length]})
    return wav_dict


def combine_dict_to_csv(phone_dur_dict, feat_dict, total_csv_path):
    files = []
    phone_dur_key = phone_dur_dict.keys()
    feat_key = feat_dict.keys()
    intersection_key = list(set(phone_dur_key) & set(feat_key))
    if len(list(phone_dur_dict.values())[0]) > 1 : 
        # {key:[phone, dur]}
        files = [(feat_dict[key][0], feat_dict[key][1], phone_dur_dict[key][1], phone_dur_dict[key][0]) for key in intersection_key]
        df = pandas.DataFrame(
                data=files, columns=["audio_feature", "wav_length_ms", "duration", "transcript"]
            )
    else: 
        # {key:[phone]}
        files = [(feat_dict[key][0], feat_dict[key][1], phone_dur_dict[key][0]) for key in intersection_key]
        df = pandas.DataFrame(
                data=files, columns=["wav_filename", "wav_length_ms", "transcript"]
            )
    # Write to CSV file which contains three columns:
    df.to_csv(total_csv_path, index=False)
    logging.info("Successfully generated csv file {}".format(total_csv_path))

def generate_features_csv(wav_dict, output_dir, trainset_config):
    """ Extract features (mel, energy, f0) and return feat dict
        feat_dict = {utt_id:{[feat_file, wav_length]}}
    """
    logging.info("Start extractor_features")
    feat_dict = {}
    featurizer = ExtractFeatures(trainset_config)
    for uttid, wav_file in wav_dict.items():
        featurizer.extractor_features(wav_file[0], output_dir)
        feat_file = os.path.join(output_dir, uttid+'.npz')
        wav_length = wav_file[1]
        if os.path.exists(feat_file):
            feat_dict.update({uttid:[feat_file, wav_length]})

    logging.info("Finished extractor_features")
    return feat_dict

def split_train_dev_test(total_csv, output_dir):
    # get total_csv
    data = pandas.read_csv(total_csv)
    headers_num = len(data.keys())
    x, y = data.iloc[:, :headers_num-1], data.iloc[:, headers_num-1:]
    x_train, x_rest, y_train, y_rest = train_test_split(
        x, y, test_size=0.1, random_state=0)
    x_test, x_dev, y_test, y_dev = train_test_split(
        x_rest, y_rest, test_size=0.9, random_state=0)
    # add ProsodyLabel
    x_train.insert(headers_num-1, 'transcript', y_train)
    x_test.insert(headers_num-1, 'transcript', y_test)
    x_dev.insert(headers_num-1, 'transcript', y_dev)
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


def processor(data_dir, output_dir, p):
    """ download and process """
    # already downloaded data_baker
    data_baker = os.path.join(data_dir, "BZNSYP")
    if os.path.exists(data_baker):
        logging.info("{} already exist".format(data_baker))
    else:
        download_and_extract(data_dir, URL)
    os.makedirs(output_dir, exist_ok=True)
    # process wav
    logging.info("ProcessingA audio for {}".format("all_files"))
    wav_dict = process_wav(data_dir)
    total_csv_path = os.path.join(data_dir, "total.csv")
    model_type = p.model
    if model_type != "fastspeech2":
        # mel-spec,f0 and energy will be extracted and saved in output_dir
        phone_dict = process_prosody(data_dir)
        combine_dict_to_csv(phone_dict, wav_dict, total_csv_path)
        split_train_dev_test(total_csv_path, data_dir)
    else:
        logging.info("Processing the phonelabeling in {}".format(data_dir))
        phone_dur_dict = process_phonelabeling(data_dir)
        feat_dict = generate_features_csv(wav_dict, output_dir, p.trainset_config)
        combine_dict_to_csv(phone_dur_dict, feat_dict, total_csv_path)
        split_train_dev_test(total_csv_path, data_dir)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        logging.warning('Usage: python {} config_json_file data_dir'.format(sys.argv[0]))
        sys.exit()
    jsonfile = sys.argv[1]
    data_dir = sys.argv[2]
    with open(jsonfile) as file:
        config = json.load(file)
    
    output_dir = os.path.join(data_dir, 'dump_feats')
    p = parse_config(config)
    processor(data_dir, output_dir, p)
