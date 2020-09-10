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


int loadFrozenGraph(tensorflow::Session *&session, 
                const std::string &graph_path, tensorflow::Status &status) {
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
    return 0;
}

tensorflow::Tensor getAsrInputFromFile(const std::string &filename) {
    // get 40-dim fbank
    std::ifstream fin(filename);
    std::vector<float> dataPoints;
    float item = 0.0;
    while (fin >> item) {
        dataPoints.push_back(item);
    }
    fin.close();

    int nrows = int(dataPoints.size() / 40);
    int ncols = 40;
    tensorflow::Tensor feats(tensorflow::DT_FLOAT, 
                             tensorflow::TensorShape({1, nrows, ncols, 1}));
    auto feats_tensor = feats.tensor<float, 4>();

    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            feats_tensor(0, row, col, 0) = dataPoints[row * 40 + col];
        }
    }
    return feats;
}

Eigen::Tensor<float, 1> computeLogSoftmax(tensorflow::Tensor score) {
    // logsoftmax = logits - log(reduce_sum(exp(logits)))
    auto score_tensor = score.tensor<float, 2>();
    int classes = score.shape().dim_size(1);
    Eigen::Tensor<float, 1> score_reshape(classes);
    for (int i = 0; i < classes; i++) {
        score_reshape(i) = score_tensor(0, i);
    }
    Eigen::Tensor<float, 0> log_sum = score_reshape.exp().sum().log();
    auto log_probs = score_reshape - score_reshape.constant(log_sum());
    return log_probs;
}

void createInputStructureEncoder(std::vector
                                 <std::pair<std::string, tensorflow::Tensor>> &inputs) {
    // input_seq: [None, input_seq_len, 40, 1]
    tensorflow::Tensor input_seq(tensorflow::DT_FLOAT);
    tensorflow::Tensor input_len(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
    inputs.emplace_back(std::pair<std::string, tensorflow::Tensor>
                        {"deploy_encoder_input_seq", input_seq});
    inputs.emplace_back(std::pair<std::string, tensorflow::Tensor>
                        {"deploy_encoder_input_length", input_len});
}

void createOutputNameStructureEncoder(std::vector<std::string> &output_names) {
    /*
        encoder_output: [1, input_seq_len // 4, hsize] float
        input_mask: [1, 1, 1, input_seq_len // 4] float
    */
    output_names.emplace_back(
    "transformer_encoder/transformer_encoder_layer_11/layer_normalization_23/batchnorm/add_1");
    output_names.emplace_back("strided_slice_1");
}

void createInputStructureDecoder(std::vector
                                 <std::pair<std::string, tensorflow::Tensor>> &inputs) {
    /*
        encoder_output: [1, input_seq_len // 4, hsize]
        memory_mask: [1, 1, 1, input_seq_len // 4]
        history_predictions: [1, pred_seq_len]
    */
    tensorflow::Tensor encoder_output(tensorflow::DT_FLOAT);
    tensorflow::Tensor memory_mask(tensorflow::DT_FLOAT);
    tensorflow::Tensor history_predictions(tensorflow::DT_FLOAT);
    tensorflow::Tensor step(tensorflow::DT_INT32, tensorflow::TensorShape({1}));

    inputs.emplace_back(std::pair<std::string, tensorflow::Tensor>
                        {"deploy_decoder_encoder_output", encoder_output});
    inputs.emplace_back(std::pair<std::string, tensorflow::Tensor>
                        {"deploy_decoder_memory_mask", memory_mask});
    inputs.emplace_back(std::pair<std::string, tensorflow::Tensor>
                        {"deploy_decoder_history_predictions", history_predictions});
    inputs.emplace_back(std::pair<std::string, tensorflow::Tensor>
                        {"deploy_decoder_step", step});
}

void createOutputNameStructureDecoder(std::vector<std::string> &output_names) {
    // logits: [1, classes] float
    output_names.emplace_back("strided_slice_3");
}

int loadSavedModel(tensorflow::SavedModelBundle &bundle, const std::string &model_path) {
    tensorflow::SessionOptions sess_options;
    tensorflow::RunOptions run_options;
    tensorflow::Status status;
    status = tensorflow::LoadSavedModel(sess_options, run_options, \
        model_path, {tensorflow::kSavedModelTagServe}, &bundle);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }
    return 0;
}

tensorflow::Tensor getTtsInputFromFile(const std::string filename) {
    std::ifstream fin(filename);
    std::vector<int> dataPoints;
    float item = 0.0;
    while (fin >> item) {
        dataPoints.push_back(item);
    }
    fin.close();

    int nrows = int(dataPoints.size());
    tensorflow::Tensor feats(tensorflow::DT_INT32, 
                             tensorflow::TensorShape({1, nrows}));
    auto feats_tensor = feats.tensor<int, 2>();

    for (int row = 0; row < nrows; row++) {
        feats_tensor(0, row) = dataPoints[row];
    }
    return feats;
}

void writeTtsOutputToFile(tensorflow::Tensor output, const std::string filename) {
    std::ofstream fout(filename);
    auto output_tensor = output.tensor<float, 3>();
    int nrows = output.shape().dim_size(1);
    int ncols = output.shape().dim_size(2);
    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            fout << output_tensor(0, row, col) << " ";
        }
        fout << "\n";
    }
    fout.close();
}

void createInputStructure(std::vector
                             <std::pair<std::string, tensorflow::Tensor>> &inputs) {
    // input: [batch_size, input_len]
    tensorflow::Tensor input(tensorflow::DT_FLOAT);
    inputs.emplace_back(std::pair<std::string, tensorflow::Tensor>
                        {"serving_default_input:0", input});
   
}

void createOutputNameStructure(std::vector<std::string> &output_names) {
    // output: [batch_size, frame_size, dim]
    output_names.emplace_back("StatefulPartitionedCall:0");
}
