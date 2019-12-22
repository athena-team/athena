# reference to https://github.com/kpu/kenlm
cd tools
wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz

mkdir -p kenlm/build
cd kenlm/build
cmake ..
make -j 4
