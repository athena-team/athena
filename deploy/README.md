# Deploying TensorFlow 2.0 models using C++

Here we describe how to deploy a TensorFlow model trained on Athena on servers, using C++ codes only. The implementation mainly replies on the TensorFlow C++ API. Firstly, we download/compile all environments that we will use. Second, we should "freeze" TensorFlow model trained on Python to pb format, which gather graph defination and weights together into a single file. Thirdly, we write C++ scripts to load pb file, and provide CMakeLists.txt to include all dependencies needed during compiling. Finally, compile the C++ codes and the executable file will be generated.

Below we will show you some example scipts to achieve this functionality step by step.


## Environment

- Operating System: Tested on CentOS Linux 7
- GCC: 7.3.0
- Bazel: 0.26.0
- TensorFlow: 2.0
- Git: 2.22.0 (should be >= 1.8)


## Step 1. Getting the TensorFlow C++ API and Other Dependencies

### (1) Get Bazel to compile TensorFlow from source
```
$ wget https://github.com/bazelbuild/bazel/releases/download/0.26.0/bazel-0.26.0-installer-linux-x86_64.sh
$ bash bazel-0.26.0-installer-linux-x86_64.sh --prefix=/some/path/bazel-0.26.0
```

### (2) Get source code of Tensorflow from GitHub
```
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout r2.0
```

### (3) Build TensorFlow from source using Bazel
```
$ export PATH=/some/path/bazel-0.26.0:$PATH
$ ./configure
```
After configuration, run the following command to compile TensorFlow:
```
$ bazel build //tensorflow:libtensorflow_cc.so
```
It may takes about an hour to compile TensorFlow. If you successfully compiled TensorFlow, you will see files such as "bazel-bin", "bazel-genfiles" and so on appear in the root of TensorFlow directory. Theses files will be used later.

### (4) Get Protobuf and absl
```
$ ./tensorflow/contrib/makefile/download_dependencies.sh
$ ./tensorflow/contrib/makefile/compile_linux_protobuf.sh
```

### (5) Get Eigen3
```
$ git clone https://gitlab.com/libeigen/eigen.git
```

### (6) Get kenlm (optional)

Now we leave a todo here. kenlm will be used if you want to load a n-gram lm.


## Step 2. Freezing TensorFlow Model
```
$ cd /path/to/athena
$ python athena/deploy_main.py examples/asr/hkust/configs/mtl_transformer_sp.json pb_path
$ cp -r pb_path deploy/graph
```

## Step 3. Loading TensorFlow Model using C++

Look at src/\*.cpp, include/\*.h and CMakeLists.txt for details.

## Step 4. Compiling the C++ Codes and Running the executable file
Firstly, we should link all dependencies to the dependencies directory:
```
$ cd /path/to/athena/deploy
$ mkdir -p dependencies
$ ln -sf /absolute/path/to/tensorflow ./dependencies/tensorflow
$ ln -sf /absolute/path/to/eigen ./dependencies/eigen
```
Then, compile the C++ Codes: 
```
$ mkdir -p build
$ cd build
$ cmake ..
$ make
```
After compiling, you can run the executable file:
```
$ ./asr
```