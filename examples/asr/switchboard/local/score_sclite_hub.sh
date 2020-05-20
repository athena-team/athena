#!/bin/bash
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

stage=0
log_file=$1
spm_model_file=$2
score_dir=$3
sctk_path=$4
stm=${score_dir}/stm
ctm=${score_dir}/ctm
hubscr=${sctk_path}/bin/hubscr.pl
hubdir=$(dirname ${hubscr})


if [ ${stage} -le 0 ]; then
    tr '\n' ' ' < ${log_file} | sed 's|INFO:absl:|\nINFO:absl:|g' | \
        grep 'INFO:absl:predictions' > ${score_dir}/decode.list

    # generate key list for scoring
    tail -n +2 examples/asr/switchboard/data/hub500.csv | sort -nk 2 | \
        awk -F "/" '{print $NF}' | awk -F "." '{print $1}' > ${score_dir}/hub500.key

    # decode bpe pieces' ids into words and reformat results
    python examples/asr/switchboard/local/result2ctm.py ${score_dir}/decode.list \
        ${score_dir}/hub500.key ${spm_model_file} ${ctm}
fi


if [ ${stage} -le 1 ]; then
    # Note: similar to kaldi
    # Remove some stuff we don't want to score, from the ctm.
    # the big expression in parentheses contains all the things that get mapped
    # by the glm file, into hesitations.
    # The -$ expression removes partial words.
    # the aim here is to remove all the things that appear in the reference as optionally
    # deletable (inside parentheses), as if we delete these there is no loss, while
    # if we get them correct there is no gain.
    cp ${ctm} ${score_dir}/tmpf;
    cat ${score_dir}/tmpf | grep -i -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
    grep -i -v -E '<UNK>' | \
    grep -i -v -E ' (UH|UM|EH|MM|HM|AH|HUH|HA|ER|OOF|HEE|ACH|EEE|EW)$' | \
    grep -v -- '-$' > ${ctm};
fi


if [ ${stage} -le 2 ]; then
    # scoring only the swbd part in hub500
    swbd_stm=${score_dir}/ref.swbd.stm
    swbd_ctm=${score_dir}/hyp.swbd.ctm
    grep -v '^en_' ${stm} > ${swbd_stm}
    grep -v '^en_' ${ctm} > ${swbd_ctm} 
    ${hubscr} -p ${hubdir} -V -l english -h hub5 -g ${score_dir}/glm -r ${swbd_stm} ${swbd_ctm} > ${score_dir}/score.swbd.log
fi
