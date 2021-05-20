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
#!/bin/bash
# Tool to convert sph audio format to wav format.
cd tools
if [ ! -e sph2pipe_v2.5.tar.gz ]; then
  wget -T 10 -t 3 http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz || exit 1
fi

rm -rf sph2pipe_v2.5 sph2pipe
tar -zxvf sph2pipe_v2.5.tar.gz || exit 1
mv sph2pipe_v2.5 sph2pipe
cd sph2pipe
gcc -o sph2pipe  *.c -lm || exit 1
