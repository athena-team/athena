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

if [ $# -ne 3 ] ; then
  echo "You should run this script with three param: 1、decode log file, 2、vocab file,\
        3、dir for saving score"
  exit 1
fi

log_file=$1
vocab_file=$2
score_dir=$3

if [ -d "$score_dir" ]; then
  rm -rf "$score_dir"
fi
mkdir "$score_dir"

tr '\n' ' ' < "$log_file" > "$score_dir"/tmp1.list || exit 1
sed 's|INFO:absl:|\nINFO:absl:|g' "$score_dir"/tmp1.list > "$score_dir"/tmp2.list || exit 1
grep 'INFO:absl:predictions' "$score_dir"/tmp2.list > "$score_dir"/decode.list || exit 1

python athena/tools/split_hyp_ref.py --decode_result "$score_dir"/decode.list \
        --vocab_dir "$vocab_file" || exit 1
rm "$score_dir"/*.list

sclite -i wsj -r "$score_dir"/ref.txt -h  "$score_dir"/hyp.txt -e utf-8 -o all -O "$score_dir" \
        > "$score_dir"/sclite.compute.log || exit 1

echo "saved sclite score in ${score_dir}:"
grep Sum/Avg "$score_dir"/hyp.txt.sys
