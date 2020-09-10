/* Copyright (C) 2020 ATHENA AUTHORS;
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
============================================================================== */
/* This is an example script to load ASR model using TensorFlow C++ API. */

#include "../include/tensor_utils.h"
#include "../include/utils.h"


int main(int argc, char** argv) {
    std::cout << "Loading model ..." << std::endl;

    // Initialization of Tensorflow session for encoder graph and decoder graph
    tensorflow::Status status;
    tensorflow::Session* enc_session;
    std::string enc_graph_path = "../graph_asr/graph/encoder.pb";
    loadFrozenGraph(enc_session, enc_graph_path, status);
    tensorflow::Session* dec_session;
    std::string dec_graph_path = "../graph_asr/graph/decoder.pb";
    loadFrozenGraph(dec_session, dec_graph_path, status);

    // Initialization of encoder and decoder inputs
    std::vector <std::pair<std::string, tensorflow::Tensor>> enc_inputs;
    createInputStructureEncoder(enc_inputs);
    std::vector <std::pair<std::string, tensorflow::Tensor>> dec_inputs;
    createInputStructureDecoder(dec_inputs);

    // Initialization of encoder and decoder outputs names
    std::vector <std::string> enc_output_names;
    createOutputNameStructureEncoder(enc_output_names);
    std::vector <std::string> dec_output_names;
    createOutputNameStructureDecoder(dec_output_names);

    // Initialization of encoder and decoder outputs values
    std::vector<tensorflow::Tensor> enc_outputs;
    std::vector<tensorflow::Tensor> dec_outputs;

    // Initialization of index_to_char map
    std::vector<std::string> index_to_char;
    createMap(index_to_char, "../graph_asr/test_data/vocab.txt");

    std::cout << "Start argmax decoding ... " << std::endl;
    auto start = std::chrono::system_clock::now();

    // Encoder part
    // 1. Place value into input_seq
    enc_inputs[0].second = getAsrInputFromFile("../graph_asr/test_data/feats.txt");

    // 2. Place value into input_len
    auto input_len = enc_inputs[1].second.tensor<int, 1>();
    input_len(0) = enc_inputs[0].second.shape().dim_size(1);

    // 3. Run Session
    status = enc_session->Run(enc_inputs, enc_output_names, {}, &enc_outputs);

    // Decoder part
    // Do argmax decoding
    // 1. Initialization, place value into dec_inputs
    bool stop = false;
    int sos_eos = index_to_char.size();
    int max_seq_len = enc_outputs[0].shape().dim_size(1);
    int max_score_index = -1; 
    int item = 0;
    float max_score = -10000;
    std::vector <int> completed_seqs;
    completed_seqs.push_back(sos_eos);
    int completed_seqs_len = completed_seqs.size();

    dec_inputs[0].second = enc_outputs[0];
    dec_inputs[1].second = enc_outputs[1];
    tensorflow::Tensor init_history_predictions(tensorflow::DT_FLOAT,
                                                tensorflow::TensorShape({1, 1}));
    auto history_predictions = init_history_predictions.tensor<float, 2>();
    history_predictions(0, 0) = float(sos_eos);
    dec_inputs[2].second = init_history_predictions;
    auto step = dec_inputs[3].second.tensor<int, 1>();
    step(0) = 1;

    // 2. Decode loop
    while(!stop) {
        status = dec_session->Run(dec_inputs, dec_output_names, {}, &dec_outputs);
        
        // compute log softmax
        auto log_probs = computeLogSoftmax(dec_outputs[0]);

        // find the largest score index
        for (int i = 0; i < sos_eos + 1; i++) {
            if (log_probs(i) > max_score) {
                max_score = log_probs(i);
                max_score_index = i;
            }
        }

        // update history_prediction and step
        completed_seqs.push_back(max_score_index);
        completed_seqs_len = completed_seqs.size();
        tensorflow::Tensor update_history_predictions(
                                    tensorflow::DT_FLOAT, 
                                    tensorflow::TensorShape({1, completed_seqs_len}));
        auto last_history_predictions = update_history_predictions.tensor<float, 2>();
        for (int i = 0; i < completed_seqs_len; i++) {
            last_history_predictions(0, i) = float(completed_seqs[i]);
        }
        dec_inputs[2].second = update_history_predictions;
        step(0) += 1;

        // judge should stop or not
        // stop if seq_len exceeds max_len or meets eos
        if (completed_seqs_len > max_seq_len || max_score_index == sos_eos) {
            stop = true;
        }

        max_score_index = -1; 
        max_score = -10000;
        dec_outputs.clear();
    }
    enc_outputs.clear();

    // 3. Print decode results
    std::string completed_seqs_char = "";
    for (int i = 0; i < completed_seqs.size(); i++) {
        item = completed_seqs[i];
        if (item != sos_eos && item != 0) {
            completed_seqs_char += index_to_char[item];
        }
    }
    std::cout << "Argmax decoding results: " << completed_seqs_char << std::endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total run time of samples: " << diff.count() << " seconds." << std::endl;
    return 0;
}
