import os
import sys
import codecs
import pandas
from absl import logging

import tensorflow as tf
from athena import get_wave_file_length


SUBSETS = ['train', 'dev', 'test']
DATASETS = ['aidatatang', 'magic_data', 'primewords', 'ST-CMDS']
def convert_audio_and_split_transcript(directory, subset, out_csv_file, DATASET):
    if DATASET not in DATASETS:
        raise ValueError(DATASET, 'is not in DATASETS')
    gfile = tf.compat.v1.gfile
    logging.info("Processing audio and transcript for {}_{}".format(DATASET, subset))
    # Prepare aidatatang data to csv
    if DATASET == 'aidatatang':

        audio_dir = os.path.join(directory, 'corpus', subset)
        trans_dir = os.path.join(directory, 'transcript')

        files = []
        char_dict = {}
        if not gfile.Exists(os.path.join(directory, subset)):
            data_file = os.path.join(directory, subset)
            # os.mkdir(data_file)
            os.makedirs(data_file, exist_ok=True)
            for filename in os.listdir(audio_dir):
                os.system("tar -zxvf" + audio_dir + filename + " -C" + data_file)

        with codecs.open(os.path.join(trans_dir, "aidatatang_200_zh_transcript.txt"), "r", encoding="utf-8") as f:
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
        output_wav_dir = os.path.join(directory, subset)

        for root, subdirs, _ in gfile.Walk(output_wav_dir):
            for subdir in subdirs:
                for filename in os.listdir(os.path.join(root, subdir)):
                    if filename.strip().split('.')[-1] != 'wav':
                        continue
                    files_size_dict[filename] = (
                        get_wave_file_length(os.path.join(root, subdir, filename)),
                        subdir,
                    )
        content = []
        for wav_filename, trans in files:
            if wav_filename in files_size_dict:
                filesize, subdir = files_size_dict[wav_filename]
                abspath = os.path.join(output_wav_dir, subdir, wav_filename)
                content.append((abspath, filesize, trans, subdir))

    # Prepare magic_data data to csv
    elif DATASET == 'magic_data':
        audio_dir = os.path.join(directory, subset)
        trans_dir = os.path.join(directory, subset, "TRANS.txt")

        files = []
        with codecs.open(trans_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                items = line.strip().split('\t')
                wav_filename = items[0]
                labels = items[2]
                speaker = items[1]
                files.append((wav_filename, speaker, labels))
        files_size_dict = {}
        for root, subdirs, _ in gfile.Walk(audio_dir):
            for subdir in subdirs:
                for filename in os.listdir(os.path.join(root, subdir)):
                    files_size_dict[filename] = (
                        get_wave_file_length(os.path.join(root, subdir, filename)),
                        subdir,
                    )
        content = []
        for wav_filename, speaker, trans in files:
            if wav_filename in files_size_dict:
                filesize, subdir = files_size_dict[wav_filename]
                abspath = os.path.join(audio_dir, subdir, wav_filename)
                content.append((abspath, filesize, trans, speaker))

    # Prepare primewords data to csv
    elif DATASET == 'primewords':
        audio_dir = os.path.join(directory, "audio_files")
        trans_dir = os.path.join(directory, "set1_transcript.json")

        files = []
        with codecs.open(trans_dir, 'r', encoding='utf-8') as f:
            items = eval(f.readline())
            for item in items:
                wav_filename = item['file']
                # labels = item['text']

                labels = ''.join([x for x in item['text'].split()])
                # speaker = item['user_id']
                files.append((wav_filename, labels))

        files_size_dict = {}
        for subdir in os.listdir(audio_dir):
            for root, subsubdirs, _ in gfile.Walk(os.path.join(audio_dir, subdir)):
                for subsubdir in subsubdirs:
                    for filename in os.listdir(os.path.join(root, subsubdir)):
                        files_size_dict[filename] = (
                            os.path.join(root, subsubdir, filename),
                            root,
                            subsubdir
                        )
        content = []
        for wav_filename, trans in files:
            if wav_filename in files_size_dict:
                filesize, root, subsubdir = files_size_dict[wav_filename]
                abspath = os.path.join(root, subsubdir, wav_filename)
                content.append((abspath, filesize, trans, None))

    # Prepare ST-CMDS data to csv
    else:
        content = []
        for filename in os.listdir(directory):
            if filename.split('.')[-1] == 'wav':
                trans_file = os.path.join(directory, filename.split('.')[0] + '.txt')
                metadata_file = os.path.join(directory, filename.split('.')[0] + '.metadata')
                with codecs.open(trans_file, 'r', encoding='utf-8') as f:
                    labels = f.readline()
                with codecs.open(metadata_file, 'r', encoding='utf-8') as f:
                    speaker = f.readlines()[2].split()[-1]
                abspath = os.path.join(directory, filename)
                filesize = get_wave_file_length(abspath)
                content.append((abspath, filesize, labels, speaker))

    files = content
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript", "speaker"]
    )
    df.to_csv(out_csv_file, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(out_csv_file))


def processor(dircetory, subset, force_process, DATASET='aidatatang'):
    """ download and process"""
    if subset not in SUBSETS:
        raise ValueError(subset, 'is not in aidatatang')
    if force_process:
        logging.info("force process is set to be true")

    subset_csv = os.path.join(dircetory, DATASET + subset + ".csv")
    if not force_process and os.path.exists(subset_csv):
        logging.info("{} already exist".format(subset_csv))
        return subset_csv
    dircetory = os.path.join(dircetory, DATASET)
    logging.info("Processing the {} subset {} in {}".format(DATASET, subset, dircetory))
    convert_audio_and_split_transcript(dircetory, subset, subset_csv, DATASET)
    logging.info("Finished processing {} subset {}".format(DATASET, subset))
    return subset_csv

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    DIR = sys.argv[1]
    for DATASET in DATASETS:
        if DATASET == 'aidatatang':
            SUBSETS = ['train', 'dev', 'test']
        elif DATASET == 'magic_data':
            SUBSETS = ['train', 'dev']
        else:
            # Primewords and ST-CDMS dataset don't split train dev test
            SUBSETS = ['train']
        for SUBSET in SUBSETS:
            processor(DIR, SUBSET, True, DATASET)

    # TODO Need merge all csv to one