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

#include "tensor_utils.h"
#include "tensorflow/core/public/session.h"
#include <fstream>


int initSession(tensorflow::Session *&session, const std::string &graph_path, tensorflow::Status &status) {
    status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }

    //Define Tensorflow graph
    tensorflow::GraphDef graph_def;
    status = ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }
}

tensorflow::Tensor getMatrixFromFile(const std::string &filename) {
    std::ifstream fin(filename);

    std::vector<float> dataPoints;
    float item = 0.0;
    while (fin >> item) {
        dataPoints.push_back(item);
    }
    int nrows = int(dataPoints.size() / 40);
    int ncols = 40;
    tensorflow::Tensor res(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, nrows, ncols, 1}));
    auto X = res.tensor<float, 4>();

    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            X(0, row, col, 0) = dataPoints[row * 40 + col];
        }
    }
    fin.close();
    return res;
}


void createInputStructureEncoder(std::vector <std::pair<std::string, tensorflow::Tensor>> &inputs){
    // input
    tensorflow::Tensor input_seq(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, 593, 40, 1}));
    tensorflow::Tensor input_len(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
    inputs.emplace_back(std::pair<std::string, tensorflow::Tensor>{"input_1", input_seq});
    inputs.emplace_back(std::pair<std::string, tensorflow::Tensor>{"input_2", input_len});

    auto b = inputs[1].second.tensor<int, 1>();
    b(0) = 593;
}

void createOutputNameStructureEncoder(std::vector<std::string> &output_names){
    /*
        encoder_output: [1, seq_length, hsize]
        input_mask: [1, 1, 1, seq_length]
    */
    output_names.emplace_back("transformer_encoder/transformer_encoder_layer_11/layer_normalization_23/batchnorm/add_1");
    output_names.emplace_back("strided_slice_1");
}
