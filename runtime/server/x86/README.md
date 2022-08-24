# Athena Server (x86) ASR Demo

## Run with Server Build
* Step 1. Download or prepare your pretrained model.
* Step 2. Build as in `Run with Local Build`
* Step 3. Start WebSocket server.

``` sh
export GLOG_logtostderr=1
export GLOG_v=2
model_dir=your_model_dir
./build/server_main 
    --port 10086 
    --threads 10
    --cmvn $model_dir/cmvn.txt
    --am_path $model_dir/model
    --dict_path $model_dir/words.txt 2>&1 | tee server.log
```
* Step 4. Start WebSocket client.

```sh
export GLOG_logtostderr=1
export GLOG_v=2
wav_path=your_test_wav_path
./build/client_main 
    --url ws://127.0.0.1:10086
    --wav_path $wav_path 2>&1 | tee client.log
```

You can also start WebSocket client by web browser as described before.

Here is a demo for command line based websocket server/client interaction.
