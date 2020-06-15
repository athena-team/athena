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
/* This is an example script to load pb using TensorFlow C++ API. */

#include "tensorflow/core/public/session.h"
#include "Eigen/Dense"

#include "../include/tensor_utils.h"
#include "../include/utils.h"


int main(int argc, char** argv) {
    std::cout << "Preparation..." << std::endl;

    // Initialization of Tensorflow session
    tensorflow::Session* session;
    tensorflow::Status status;
    std::string graph_path = "../graph/encoder.pb";
    initSession(session, graph_path, status);

    // Initialization of encoder inputs
    std::vector <std::pair<std::string, tensorflow::Tensor>> inputs;
    createInputStructureEncoder(inputs);

    // Initialization of encoder outputs names
    std::vector <std::string> output_names;
    createOutputNameStructureEncoder(output_names);

    // Initialization of encoder outputs values
    std::vector<tensorflow::Tensor> outputs;

    // Initialization of index_to_char map
    std::vector<std::string> index_to_char;
    createMap(index_to_char, "../input_feats/char3650.txt");
    std::cout << index_to_char.size() << std::endl;

    // Run for the first time to prevent initialization overhead
    status = session->Run(inputs, output_names, {}, &outputs);
    outputs.clear();

    std::cout << "Start!" << std::endl;
    auto start = std::chrono::system_clock::now();
    
    // place value into input
    inputs[0].second = getMatrixFromFile("../input_feats/feats.txt");
    std::cout << inputs[0].second.shape() << std::endl;

    // Run Session
    status = session->Run(inputs, output_names, {}, &outputs);

    std::cout << outputs[0].shape() << std::endl;
    std::cout << outputs[1].shape() << std::endl;

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total run time of samples: " << diff.count() << " seconds." << std::endl;
}
