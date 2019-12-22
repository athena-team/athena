#!/bin/bash
# Tool to convert sph audio format to wav format.
cd tools
if [ ! -e sph2pipe_v2.5.tar.gz ]; then
  wget -T 10 -t 3 http://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz || exit 1
fi

tar -zxvf sph2pipe_v2.5.tar.gz || exit 1
cd sph2pipe_v2.5/
gcc -o sph2pipe  *.c -lm || exit 1
