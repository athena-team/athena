#
#!/usr/bin/env bash
# coding=utf-8
# Copyright (C) 2021 ATHENA AUTHORS; Shuaijiang Zhao; Xiaoning Lei
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
# reference https://github.com/SpeechColab/GigaSpeech/tree/main/utils

import os
import re
import sys
import json
from absl import logging

SUBSETS = ["XL", "DEV", "TEST"]
garbage_utterance_tags = "<SIL>|<MUSIC>|<NOISE>|<OTHER>"
punctuation_tags = "<COMMA>|<EXCLAMATIONPOINT>|<PERIOD>|<QUESTIONMARK>"


def extract_json(json_file='', output_dir=''):
    input_dir = os.path.dirname(json_file)
    try:
        with open(json_file, 'r') as JSONFILE:
            json_data = json.load(JSONFILE)
    except:
        sys.exit(f'Failed to load input json file: {json_file}')
    else:
        if json_data['audios'] is not None:
            with open(f'{output_dir}/utt2subsets', 'w') as utt2subsets, \
                    open(f'{output_dir}/text', 'w') as utt2text, \
                    open(f'{output_dir}/segments', 'w') as segments, \
                    open(f'{output_dir}/opus.scp', 'w') as wavscp:
                for long_audio in json_data['audios']:
                    try:
                        long_audio_path = os.path.realpath(
                            os.path.join(input_dir, long_audio['path']))
                        aid = long_audio['aid']
                        segments_lists = long_audio['segments']
                        assert (os.path.exists(long_audio_path))
                        assert ('opus' == long_audio['format'])
                        assert (16000 == long_audio['sample_rate'])
                    except AssertionError:
                        print(f'Warning: {aid} something is wrong, maybe AssertionError, skipped')
                        continue
                    except:
                        print(f'Warning: {aid} something is wrong, maybe the error path: '
                              f'{long_audio_path}, skipped')
                        continue
                    else:
                        wavscp.write(f'{aid}\t{long_audio_path}\n')
                        for segment_file in segments_lists:
                            try:
                                sid = segment_file['sid']
                                start_time = segment_file['begin_time']
                                end_time = segment_file['end_time']
                                text = segment_file['text_tn']
                                segment_subsets = segment_file["subsets"]
                            except:
                                print(f'Warning: {segment_file} something is wrong, skipped')
                                continue
                            else:
                                utt2text.write(f'{sid}\t{text}\n')
                                segments.write(f'{sid}\t{aid}\t{start_time}\t{end_time}\n')
                                segment_sub_names = " ".join(segment_subsets)
                                utt2subsets.write(f'{sid}\t{segment_sub_names}\n')


def convert_opus2wav(opus_scp='', wav_scp='', rm_opus=False):
    with open(opus_scp, 'r') as oscp, open(wav_scp, 'w') as wscp:
        for line in oscp:
            line = line.strip()
            utt, opus_path = re.split('\s+', line)
            wav_path = opus_path.replace('.opus', '.wav')
            cmd = f'ffmpeg -y -i {opus_path} -ac 1 -ar 16000 {wav_path}'
            try:
                os.system(cmd)
                wscp.write(f'{utt}\t{wav_path}\n')
            except:
                sys.exit(f'Failed to run the cmd: {cmd}')

            if rm_opus is True:
                os.remove(opus_path)

def prepare_data(data_dir='', subset='XL'):
    subset_file = os.path.join(data_dir, 'utt2subsets')
    text_file = os.path.join(data_dir, 'text')
    segment_file = os.path.join(data_dir, 'segments')
    wav_scp = os.path.join(data_dir, 'wav.scp')
    out_f = os.path.join(data_dir, subset + '.csv')

    subset_dict = {}
    with open(subset_file) as SUBSET:
        subset_lines = SUBSET.readlines()
    for line in subset_lines:
        line_list = line.strip().split()
        utt_key = line_list[0]
        subset_dict[utt_key] = line_list[1:]

    with open(text_file) as TEXT:
        text_lines = TEXT.readlines()

    time_d = {}
    with open(segment_file) as SEGMENT:
        seg_lines = SEGMENT.readlines()
    for i in seg_lines:
        item = i.strip().split('\t')
        utt_key = item[0]
        start_time = item[2]
        end_time = item[3]
        time_d[utt_key] = str(int((float(end_time) - float(start_time)) * 1000))

    text_d = {}
    for i in text_lines:
        utt_key = i.split('\t')[0]
        speaker, k1 = utt_key.split('_')
        if speaker not in text_d:
            text_d[speaker] = []
        transcriptions = i.split(utt_key)[1].strip()
        if utt_key in time_d:
            if re.search(garbage_utterance_tags, transcriptions):
                continue
            if '{' + subset + '}' not in subset_dict[utt_key]:
                continue
            # remove the punctuation tags
            transcriptions = re.sub(punctuation_tags, "", transcriptions)
            # convert two spaces to one space
            transcriptions = re.sub("  ", "", transcriptions)
            text_d[speaker].append(utt_key + '\t' + time_d[utt_key] +
                                   '\t' + transcriptions + '\t' + speaker)

    with open(wav_scp) as f:
        lines = f.readlines()
    utt_key_wav = {}
    for i in lines:
        utt_key = i.split('\t')[0]
        if utt_key not in utt_key_wav:
            utt_key_wav[utt_key] = 0

    with open(out_f, 'w') as f:
        f.write('wav_filename\twav_len\ttranscript\tspeaker\n')
        for speaker in text_d:
            if speaker in utt_key_wav:
                for utt_sample in text_d[speaker]:
                    f.write(utt_sample + '\n')


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    if len(sys.argv) < 3:
        print('Usage: python {} dataset_dir output_dir\n'
              '    dataset_dir : directory contains GigaSpeech dataset\n'
              '    output_dir  : GigaSpeech data working directory'.format(sys.argv[0]))
        exit(1)
    DATASET_DIR = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
    json_file = os.path.join(DATASET_DIR, "GigaSpeech.json")
    extract_json(json_file=json_file, output_dir=OUTPUT_DIR)

    print(f'Converting opus to wave, please be patient')
    opus_scp = os.path.join(OUTPUT_DIR, 'opus.scp')
    wav_scp = os.path.join(OUTPUT_DIR, 'wav.scp')
    convert_opus2wav(opus_scp, wav_scp, False)

    for subset in SUBSETS:
        prepare_data(data_dir=OUTPUT_DIR, subset=subset)
