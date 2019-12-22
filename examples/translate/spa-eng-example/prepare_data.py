# Only support eager mode
# pylint: disable=invalid-name

""" An example using translation dataset """
import os
import re
import io
import sys
import unicodedata
import pandas
import tensorflow as tf
from absl import logging
from athena import LanguageDatasetBuilder


def preprocess_sentence(w):
    """ preprocess_sentence """
    w = ''.join(c for c in unicodedata.normalize('NFD', w.lower().strip())
        if unicodedata.category(c) != 'Mn')
    w = re.sub(r"([?.!,?])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,?]+", " ", w)
    w = w.rstrip().strip()
    return w

def create_dataset(csv_file="examples/translate/spa-eng-examples/data/train.csv"):
    """ create and store in csv file """
    path_to_zip = tf.keras.utils.get_file('spa-eng.zip',
        origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True)
    path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"
    lines = io.open(path_to_file, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:None]]
    df = pandas.DataFrame(
        data=word_pairs, columns=["input_transcripts", "output_transcripts"]
    )
    csv_dir = os.path.dirname(csv_file)
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)
    df.to_csv(csv_file, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(csv_file))

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    CSV_FILE = sys.argv[1]
    create_dataset(CSV_FILE)
