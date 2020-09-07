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
/* This is an example script to load TTS model using TensorFlow C++ API. */

#include "../include/tensor_utils.h"
#include "../include/utils.h"


int main(int argc, char** argv) {
    tensorflow::SavedModelBundle bundle;
    tensorflow::Status status;

    // load saved model
    std::cout << "Loading model ..." << std::endl;
    std::string model_path = "../graph_tts/saved_model";
    loadSavedModel(bundle, model_path);
    std::unique_ptr<tensorflow::Session>& sess = bundle.session;

    // initialize input
    std::vector <std::pair<std::string, tensorflow::Tensor>> inputs;
    createInputStructure(inputs);

    // initialize output
    std::vector <std::string> output_names;
    createOutputNameStructure(output_names);
    std::vector<tensorflow::Tensor> outputs;
    
    // run
    inputs[0].second = getTtsInputFromFile("../graph_tts/test_data/input.txt");
    auto start = std::chrono::system_clock::now();
    status = sess->Run(inputs, output_names, {}, &outputs);
    auto end = std::chrono::system_clock::now();
    writeTtsOutputToFile(outputs[0], "../graph_tts/test_data/output.txt");
    
    std::chrono::duration<double> diff = end - start;
    std::cout << "Total run time of samples: " << diff.count() << " seconds." << std::endl;
    return 0;
}
