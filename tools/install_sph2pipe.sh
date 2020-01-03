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
