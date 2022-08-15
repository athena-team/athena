# Athena Server (x86) ASR Demo

## Run with Local Build

* Step 1. Download or prepare your pretrained model.

* Step 2. Build. The build requires cmake 3.14 or above. For building, please first change to `athena/runtime/server/x86` as your build directory, then type:

``` sh
mkdir build && cd build && cmake .. && cmake --build .
```

* Step 3. Testing, the RTF(real time factor) is shown in the console.

``` sh
export GLOG_logtostderr=1
export GLOG_v=2
wav_scp=your_test_wav_path
model_dir=your_model_dir
./build/decoder_main 
    --wav_scp $wav_scp 
    --cmvn $model_dir/cmvn.txt
    --am_path $model_dir/model
    --dict_path $model_dir/words.txt 2>&1 | tee log.txt
```
