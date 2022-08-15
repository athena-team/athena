# coding=utf-8
# Copyright (C) ATHENA AUTHORS
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

#!/usr/bin/python
# -*- coding: utf-8 -*-

""" vad dataset """

import os
import sys
import tarfile
import zipfile
import urllib.request
from absl import logging

def download_file(dataset_root_dir, dataset, source):
    """
    Downloads url source
    Args:
        dataset_root_dir: local filepath
        dataset: name of dataset 
        source: url of resource
    """
    destination_file_type = '.tar.gz' 
    if source.endswith('.zip'):
        destination_file_type = '.zip'
    destination_file = os.path.join(dataset_root_dir, dataset + destination_file_type)
    if not os.path.exists(destination_file):
        logging.info(f"{destination_file} does not exist. Downloading ...")
        urllib.request.urlretrieve(source, filename=destination_file + '.tmp')
        os.rename(destination_file + '.tmp', destination_file)
        logging.info(f"Destination {destination_file}.")
    else:
        logging.info(f"Destination {destination_file} exists. Skipping.")
    return destination_file_type, destination_file

def extract_files(dataset_root_dir, dataset, source_file_type, source_file):
    """
    Extract files from compressed source file
    Args:
        dataset_root_dir: local filepath
        dataset: name of dataset
        source_file_type: type of compressed file, .tar.gz or .zip
        source_file: compressed source file
    """
    dataset_dir = os.path.join(dataset_root_dir, dataset)
    if not os.path.exists(dataset_dir):
        try:
            if source_file_type == '.tar.gz':
                source_f = tarfile.open(source_file)
            elif source_file_type == '.zip':
                source_f = zipfile.ZipFile(source_file)
            else:
                logging.info('{source_file_type} is not supported')
                return
            source_f.extractall(dataset_dir)
            source_f.close()
        except Exception:
            logging.info('Not extracting. Maybe already there?')
    else:
        logging.info(f'Skipping extracting. Data already there {dataset_dir}')

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        logging.info('Usage: python {} dataset_root_dir\n'
              '    dataset_root_dir : root directory of dataset \n'
              '    dataset : name of dataset \n'
              '    url : path of source'.format(sys.argv[0]))
        exit(1)
    DATASET_ROOT_DIR = sys.argv[1]
    DATASET = sys.argv[2]
    URL = sys.argv[3]
    source_file_type, source_file = download_file(DATASET_ROOT_DIR, DATASET, URL)
    extract_files(DATASET_ROOT_DIR, DATASET, source_file_type, source_file)
