#!/bin/bash
conda_dir=$1
work_dir=`pwd`

if [ $# -ne 1 ]; then
  echo $#
  echo "sh $0 conda_dir"  
  exit
fi
if [ -f horovod.env ];then
  echo You have build horovod env. If you want to rebuild it, you should '"' rm horovod.env'"' at first.
  exit
fi
function version_lt() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" != "$1"; }

function build_gcc_env()
{
  gcc_vision=`gcc -dumpversion`
  echo $gcc_vision
  if version_lt $gcc_vision 7.0 ;then
    echo " gcc $gcc_vision should be updated"
    #sudo yum -y install devtoolset-6-gcc devtoolset-6-gcc-c++ devtoolset-6-binutils    
    sudo yum install centos-release-scl
    sudo yum install devtoolset-7
    source /opt/rh/devtoolset-7/enable
  fi
}

function build_openmpi()
{
  if [ ! -f openmpi-4.0.2.tar.gz ];then
  echo  wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz
  fi
  if [ ! -f openmpi-4.0.2 ];then
    tar zxvf openmpi-4.0.2.tar.gz
    cd openmpi-4.0.2
    ./configure --prefix=$work_dir/openmpi-4.0.2/bin
    make clean
    make -j16
    make install
    cd -
  fi
}

function build_horovod()
{
  if [ ! -d horovod ];then
    source $work_dir/horovod.env
    git clone --recursive https://github.com/uber/horovod.git
    cd horovod 
    python setup.py clean
    HOROVOD_WITH_TENSORFLOW=1 python setup.py bdist
    rm -rf $conda_dir/lib/python3.6/site-packages/horovod*
    cp -rp $work_dir/horovod/build/lib.linux-x86_64-3.6/horovod $conda_dir/lib/python3.6/site-packages/
    cd -
  fi
}

function build_horovod_env()
{
  build_gcc_env
  build_openmpi
  build_horovod  
}

function gen_horovod_env_file()
{
  echo "export PATH=$conda_dir/bin:\$PATH">>horovod.env
  echo "export PATH=$work_dir/openmpi-4.0.2/bin:\$PATH">>horovod.env
  echo "export LD_LIBRARY_PATH=$work_dir/openmpi-4.0.2/lib:\$LD_LIBRARY_PATH">>horovod.env
  echo "export PATH=$work_dir/horovod/build/scripts-3.6:\$PATH">>horovod.env
}

function test()
{
 source $work_dir/horovod.env
 if [ -d $work_dir/horovod/examples ];then
   cd $work_dir/horovod/examples/ 
   horovodrun -np 1 -H localhost:1 python tensorflow2_mnist.py
 else
   echo WARNING: no such directory
 fi
}

function main()
{
  gen_horovod_env_file
  build_horovod_env
  test
}

main
