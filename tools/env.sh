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
# check if we are executing or sourcing.
if [[ "$0" == "$BASH_SOURCE" ]]
then
    echo "You must source this script rather than executing it."
    exit -1
fi

# don't use readlink because macOS won't support it.
if [[ "$BASH_SOURCE" == "/"* ]]
then
    export MAIN_ROOT=$(dirname $BASH_SOURCE)
else
    export MAIN_ROOT=$PWD/
fi

# pip bins
export PATH=$PATH:~/.local/bin

# athena
export PYTHONPATH=${PYTHONPATH}:$MAIN_ROOT:
