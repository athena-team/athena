#!/bin/bash
function version_lt() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" != "$1"; }
python_dir=`which python`
python_vision=`python --version 2>&1`
python_vision=${python_vision#* }
python_vision=${python_vision%% *}
echo $python_vision
if version_lt $python_vision 3.6 ;then
  echo Your python vision should be equal or greater than 3.6.9.
  echo We suggest you install Anaconda: https://www.anaconda.com/products/individual#linux
  echo and run '"sh build_horovod_centos_env.sh"' again.
  exit
fi
python_dir=${python_dir%/*}
site_packages_dir=`python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"`
work_dir=`pwd`
echo $python_dir
echo $site_packages_dir

if [ -f horovod.env ];then
  echo You have build horovod env. If you want to rebuild it, you should '"' rm horovod.env'"' at first.
  exit
fi

function build_gcc_env()
{
  gcc_vision=`gcc -dumpversion`
  echo $gcc_vision
  if version_lt $gcc_vision 7.0 ;then
    echo " gcc $gcc_vision should be updated"
    #sudo yum -y install devtoolset-6-gcc devtoolset-6-gcc-c++ devtoolset-6-binutils    
    #NOTE: yum issue: https://blog.csdn.net/qq_30938705/article/details/87281698
    sudo yum install centos-release-scl
    sudo yum install devtoolset-7
    source /opt/rh/devtoolset-7/enable
  fi
}

function build_openmpi()
{
  if [ ! -f openmpi-4.0.2.tar.gz ];then
    echo wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz
    wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz
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
    git clone --recursive https://github.com/8z8z8z/horovod.git
  fi
    cd horovod 
    if [ -d dist ];then
     rm -rf dist
    fi
    source /opt/rh/devtoolset-7/enable
    python setup.py clean
    HOROVOD_WITH_TENSORFLOW=1 python setup.py bdist
    horovod_dir=`readlink -f build/lib.linux-*/horovod`
    rm -rf $site_packages_dir/horovod*
    cd dist
    pwd
    tar xzvf *tar.gz 
    cd -
    mkdir $work_dir/horovod/build/bin
    cp `find |grep horovodrun` $work_dir/horovod/build/bin
    echo "cp -rp ./$site_packages_dir/horovod* $site_packages_dir/"
    cp -rp $work_dir/horovod/dist/$site_packages_dir/horovod* $site_packages_dir/
    horovod_bin_dir=$work_dir/horovod/build/bin
}

function build_horovod_env()
{
  build_gcc_env
  build_openmpi
  build_horovod  
}

function gen_openmpi_env_file()
{
  echo "export PATH=$python_dir:\$PATH">>horovod.env
  echo "export PATH=$work_dir/openmpi-4.0.2/bin:\$PATH">>horovod.env
  echo "export LD_LIBRARY_PATH=$work_dir/openmpi-4.0.2/lib:\$LD_LIBRARY_PATH">>horovod.env
}
function gen_horovod_env_file()
{
  horovod_bin_dir=$work_dir/horovod/build/bin
  echo "export PATH=$horovod_bin_dir:\$PATH">>horovod.env
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
  gen_openmpi_env_file
  build_horovod_env
  gen_horovod_env_file
  test
}

main
