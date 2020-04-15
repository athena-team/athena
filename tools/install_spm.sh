# reference to https://github.com/google/sentencepiece
cd tools
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j 4
sudo make install
sudo ldconfig -v
