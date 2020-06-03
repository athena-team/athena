# reference to https://github.com/usnistgov/SCTK
cd tools
git clone https://github.com/usnistgov/SCTK.git
cd SCTK
export CXXFLAGS="-std=c++11" && make config
make all
make check
make install
make doc
