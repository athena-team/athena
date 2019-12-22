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
""" switchboard_fisher dataset """

import os
import re
import sys
import codecs
from tempfile import TemporaryDirectory
import fnmatch
import pandas
from absl import logging

import tensorflow as tf
from sox import Transformer
from athena import get_wave_file_length

SUBSETS = ["train", "switchboard", "fisher", "hub500", "rt03s"]


def normalize_trans_swd(trans):
    """ TODO: docstring
    """

    norm_trans = trans.lower()
    # t.v.'s -> t.v's
    norm_trans = re.sub("\.'", "'", norm_trans)

    # [ex[specially]-/especially] -> ex
    new_line = []
    for word in norm_trans.split():
        if "/" in word and "[" in word:
            new_line.append(word.split("/")[0].split("[")[1])
        else:
            new_line.append(word)
    norm_trans = " ".join(new_line)

    # remove <b_aside>, <e_aside>, [vocalized-noise], [vocalized-laughter]
    # [laughter-yes] -> yes] (we'll process ] later)
    # them_1 -> them
    # remove [silence], [laughter], [noise]; ab[solute]- -> ab-
    # remove ], {yuppiedom} -> yuppiedom, 20/20 -> 20 20
    remove_set = [
        "<.*?>",
        "\[vocalized\-(noise|laughter)\]",
        "\[laughter\-",
        "_1",
        "\[.*?\]",
        "[\]\{\}/,?\._]",
    ]
    for pat in remove_set:
        norm_trans = re.sub(pat, " ", norm_trans)

    # remove - that does not followed by characters or at the first posi; 
    # remove ' that at the fist posi.
    # e.g.: ab- -> ab; -til -> til; 'cause -> cause
    new_line = []
    for word in norm_trans.split():
        new_word = re.sub("^-|-$", "", word)
        new_word = re.sub("^'", "", new_word)
        new_line.append(new_word)
    norm_trans = " ".join(new_line)
    norm_trans = " ".join(norm_trans.split())
    return norm_trans


def normalize_trans_fisher(trans):
    norm_trans = trans.lower()

    # t.v.'s -> t.v's
    norm_trans = re.sub("\.'", "'", norm_trans)
    norm_trans = re.sub("\[.*?\]|[,?\._]", " ", norm_trans)

    # remove - that does not followed by characters or at the first posi;
    # remove ' that at the fist posi.
    # e.g.: ab- -> ab; -til -> til.; 'cause -> cause
    new_line = []
    for word in norm_trans.split():
        new_word = re.sub("^-|-$", "", word)
        new_word = re.sub("^'", "", new_word)
        new_line.append(new_word)
    norm_trans = " ".join(new_line)
    norm_trans = " ".join(norm_trans.split())
    return norm_trans


def normalize_trans_hub_rts(trans):
    norm_trans = trans.lower()

    # t.v.'s -> t.v's
    norm_trans = re.sub("\.'", "'", norm_trans)
    # (QUI-) YEAH IT SHOULD BE QUICK AND PAINLESS AND (%HESITATION) (I-) ((WITH)
    # -> QUI YEAH IT SHOULD BE QUICK AND PAINLESS AND I
    norm_trans = re.sub("\(%.*?\)|\(\(.*?\)|<.*?>|\-\)|[,?\._\(\)]", " ", norm_trans)

    new_line = []
    for word in norm_trans.split():
        new_word = re.sub("^-|-$", "", word)
        new_word = re.sub("^'", "", new_word)
        new_line.append(new_word)
    norm_trans = " ".join(new_line)
    norm_trans = " ".join(norm_trans.split())
    return norm_trans


def split_line_and_norm_swd(line, filename=""):
    sph_key = ""
    speaker = ""
    time_start = 0.0
    time_end = 0.0
    norm_trans = ""
    if len(line.split()) < 4:
        return sph_key, speaker, float(time_start), float(time_end), norm_trans
    sph_trans_key, time_start, time_end, transcript = line.split(None, 3)
    speaker = sph_trans_key.split("-")[0][-1]
    sph_key = (
        sph_trans_key.split("-")[0][:2].upper() + "0" + sph_trans_key.split("-")[0][2:6]
    )
    norm_trans = normalize_trans_swd(transcript)
    return sph_key, speaker, float(time_start), float(time_end), norm_trans


def split_line_and_norm_fisher(line, filename=""):
    sph_key = ""
    speaker = ""
    time_start = 0.0
    time_end = 0.0
    norm_trans = ""
    if len(line.split(":")) < 2 or "#" in line or "((" in line:
        return sph_key, speaker, float(time_start), float(time_end), norm_trans
    info, transcript = line.split(":", 1)
    time_start, time_end, speaker = info.split(" ", 2)
    norm_trans = normalize_trans_fisher(transcript)
    sph_key = filename.split(".")[0]
    return sph_key, speaker, float(time_start), float(time_end), norm_trans


def split_line_and_norm_hub_rts(line, filename=""):
    sph_key = ""
    speaker = ""
    time_start = 0.0
    time_end = 0.0
    norm_trans = ""
    if len(line.split()) < 7 or ";;" in line or "IGNORE_TIME_SEGMENT_" in line:
        return sph_key, speaker, float(time_start), float(time_end), norm_trans
    sph_key, _, speaker_info, time_start, time_end, _, transcript = line.split(None, 6)
    if len(speaker_info.split("_")) < 3:
        return sph_key, speaker, float(time_start), float(time_end), norm_trans
    speaker = speaker_info.split("_")[2]
    norm_trans = normalize_trans_hub_rts(transcript)
    return sph_key, speaker, float(time_start), float(time_end), norm_trans


def convert_audio_and_split_transcript(directory, subset):
    """Convert SPH to WAV and split the transcript.
  Args:
      directory: the directory which holds the input dataset.
      subset: the name of the specified dataset. supports train 
        (switchboard+fisher), switchboard, fisher, hub500 and rt03s.
  """
    logging.info("Processing audio and transcript for %s" % subset)
    gfile = tf.compat.v1.gfile
    sph2pip = os.path.join(os.path.dirname(__file__), "../utils/sph2pipe")

    swd_audio_trans_dir = [os.path.join(directory, "LDC97S62")]
    fisher_audio_dirs = [
        os.path.join(directory, "LDC2004S13"),
        os.path.join(directory, "LDC2005S13"),
    ]
    fisher_trans_dirs = [
        os.path.join(directory, "LDC2004T19"),
        os.path.join(directory, "LDC2005T19"),
    ]
    hub_audio_dir = [os.path.join(directory, "LDC2002S09")]
    hub_trans_dir = [os.path.join(directory, "LDC2002T43")]
    rts_audio_trans_dir = [os.path.join(directory, "LDC2007S10")]

    if subset == "train":
        # Combination of switchboard corpus and fisher corpus.
        audio_dir = swd_audio_trans_dir + fisher_audio_dirs
        trans_dir = swd_audio_trans_dir + fisher_trans_dirs
    elif subset == "switchboard":
        audio_dir = swd_audio_trans_dir
        trans_dir = swd_audio_trans_dir
    elif subset == "fisher":
        audio_dir = fisher_audio_dirs
        trans_dir = fisher_trans_dirs
    elif subset == "hub500":
        audio_dir = hub_audio_dir
        trans_dir = hub_trans_dir
    elif subset == "rt03s":
        audio_dir = rts_audio_trans_dir
        trans_dir = rts_audio_trans_dir
    else:
        raise ValueError(subset, " is not in switchboard_fisher")

    subset_dir = os.path.join(directory, subset)
    if not gfile.Exists(subset_dir):
        gfile.MakeDirs(subset_dir)
    output_wav_dir = os.path.join(directory, subset + "/wav")
    if not gfile.Exists(output_wav_dir):
        gfile.MakeDirs(output_wav_dir)
    tmp_dir = os.path.join(directory, "tmp")
    if not gfile.Exists(tmp_dir):
        gfile.MakeDirs(tmp_dir)

    # Build SPH dict.
    files = []
    sph_files_dict = {}
    for sub_audio_dir in audio_dir:
        for root, _, filenames in gfile.Walk(sub_audio_dir):
            for filename in fnmatch.filter(filenames, "*.[Ss][Pp][Hh]"):
                sph_key = os.path.splitext(filename)[0]
                sph_file = os.path.join(root, filename)
                sph_files_dict[sph_key] = sph_file

    with TemporaryDirectory(dir=tmp_dir) as output_tmp_wav_dir:
        for sub_trans_dir in trans_dir:
            if sub_trans_dir in swd_audio_trans_dir:
                fnmatch_pat = "*-trans.text"
                split_and_norm_func = split_line_and_norm_swd
            elif sub_trans_dir in fisher_trans_dirs:
                fnmatch_pat = "*.[Tt][Xx][Tt]"
                split_and_norm_func = split_line_and_norm_fisher
            elif sub_trans_dir in hub_trans_dir:
                fnmatch_pat = "hub5e00.english.000405.stm"
                split_and_norm_func = split_line_and_norm_hub_rts
            else:
                fnmatch_pat = "*.stm"
                split_and_norm_func = split_line_and_norm_hub_rts

            for root, _, filenames in gfile.Walk(sub_trans_dir):
                for filename in fnmatch.filter(filenames, fnmatch_pat):
                    trans_file = os.path.join(root, filename)
                    if 1 in [
                        ele in root
                        for ele in [
                            "doc",
                            "DOC",
                            "mandarin",
                            "arabic",
                            "concatenated",
                            "bnews",
                        ]
                    ]:
                        continue
                    with codecs.open(trans_file, "r", "utf-8") as fin:
                        for line in fin:
                            line = line.strip()
                            (
                                sph_key,
                                speaker,
                                time_start,
                                time_end,
                                norm_trans,
                            ) = split_and_norm_func(line, filename)

                            # Too short, skip the wave file
                            if time_end - time_start <= 0.1:
                                continue
                            if norm_trans == "":
                                continue
                            if speaker == "A":
                                channel = 1
                            else:
                                channel = 2

                            # Convert SPH to split WAV.
                            if sph_key not in sph_files_dict:
                                print(sph_key + " not found, please check.")
                                continue
                            sph_file = sph_files_dict[sph_key]
                            wav_file = os.path.join(
                                output_tmp_wav_dir, sph_key + "." + speaker + ".wav"
                            )
                            if not gfile.Exists(sph_file):
                                raise ValueError(
                                    "the sph file {} is not exists".format(sph_file)
                                )

                            sub_wav_filename = "{0}-{1}-{2:06d}-{3:06d}".format(
                                sph_key,
                                speaker,
                                round(time_start * 100),
                                round(time_end * 100),
                            )
                            sub_wav_file = os.path.join(
                                output_wav_dir, sub_wav_filename + ".wav"
                            )

                            if not gfile.Exists(sub_wav_file):
                                if not gfile.Exists(wav_file):
                                    sph2pipe_cmd = (
                                        sph2pip
                                        + " -f wav -c {} -p ".format(str(channel))
                                        + sph_file
                                        + " "
                                        + wav_file
                                    )
                                    os.system(sph2pipe_cmd)
                                tfm = Transformer()
                                tfm.trim(time_start, time_end)
                                tfm.build(wav_file, sub_wav_file)
                            wav_filesize = os.path.getsize(sub_wav_file)

                            wav_length = get_wave_file_length(sub_wav_file)
                            files.append(
                                (os.path.abspath(sub_wav_file), wav_length, norm_trans)
                            )

    # Write to CSV file which contains three columns:
    # "wav_filename", "wav_length_ms", "transcript".
    out_csv_file = os.path.join(directory, subset + ".csv")
    df = pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_length_ms", "transcript"]
    )
    df.to_csv(out_csv_file, index=False, sep="\t")
    logging.info("Successfully generated csv file {}".format(out_csv_file))


def processor(dircetory, subset, force_process):
    """ process """
    if subset not in SUBSETS:
        raise ValueError(subset, "is not in switchboard_fisher")

    subset_csv = os.path.join(dircetory, subset + ".csv")
    if not force_process and os.path.exists(subset_csv):
        return subset_csv
    logging.info(
        "Processing the switchboard_fisher subset {} in {}".format(subset, dircetory)
    )
    convert_audio_and_split_transcript(dircetory, subset)
    logging.info("Finished processing switchboard_fisher subset {}".format(subset))
    return subset_csv


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    DIR = sys.argv[1]
    for SUBSET in SUBSETS:
        processor(DIR, SUBSET, True)
