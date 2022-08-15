//
// Created by serizba on 29/6/20.
//

#ifndef CPPFLOW2_MODEL_H
#define CPPFLOW2_MODEL_H

#include <tensorflow/c/c_api.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#include "context.h"
#include "defer.h"
#include "tensor.h"

namespace cppflow {

    class model {
    public:
        enum TYPE
        {
            SAVED_MODEL,
            FROZEN_GRAPH,
        };

        explicit model(const std::string& filename, const TYPE type=TYPE::SAVED_MODEL);

        std::vector<std::string> get_operations() const;
        std::vector<int64_t> get_operation_shape(const std::string& operation) const;

        std::vector<tensor> operator()(std::vector<std::tuple<std::string, tensor>> inputs, std::vector<std::string> outputs);
        tensor operator()(const tensor& input);

        ~model() = default;
        model(const model &model) = default;
        model(model &&model) = default;
        model &operator=(const model &other) = default;
        model &operator=(model &&other) = default;

    private:
        TF_Buffer * readGraph(const std::string& filename);

        std::shared_ptr<TF_Graph> graph;
        std::shared_ptr<TF_Session> session;
    };
}


namespace cppflow {

    inline model::model(const std::string &filename, const TYPE type) {
        this->graph = {TF_NewGraph(), TF_DeleteGraph};

        // Create the session.
        std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)> session_options = {TF_NewSessionOptions(), TF_DeleteSessionOptions};

        auto session_deleter = [](TF_Session* sess) {
            TF_DeleteSession(sess, context::get_status());
            status_check(context::get_status());
        };

        if (type == TYPE::SAVED_MODEL) {
            std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> run_options = {TF_NewBufferFromString("", 0), TF_DeleteBuffer};
            std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> meta_graph = {TF_NewBuffer(), TF_DeleteBuffer};

            int tag_len = 1;
            const char* tag = "serve";
            this->session = {TF_LoadSessionFromSavedModel(session_options.get(), run_options.get(), filename.c_str(),
                                    &tag, tag_len, this->graph.get(), meta_graph.get(), context::get_status()),
                            session_deleter};
        }
        else if (type == TYPE::FROZEN_GRAPH)  {
            this->session = {TF_NewSession(this->graph.get(), session_options.get(), context::get_status()), session_deleter};
            status_check(context::get_status());

            // Import the graph definition
            TF_Buffer* def = readGraph(filename);
            if(def == nullptr) {
                throw std::runtime_error("Failed to import graph def from file");
            }

            std::unique_ptr<TF_ImportGraphDefOptions, decltype(&TF_DeleteImportGraphDefOptions)> graph_opts = {TF_NewImportGraphDefOptions(), TF_DeleteImportGraphDefOptions};
            TF_GraphImportGraphDef(this->graph.get(), def, graph_opts.get(), context::get_status());
            TF_DeleteBuffer(def);
        }
        else {
            throw std::runtime_error("Model type unknown");
        }

        status_check(context::get_status());
    }

    inline std::vector<std::string> model::get_operations() const {
        std::vector<std::string> result;
        size_t pos = 0;
        TF_Operation* oper;

        // Iterate through the operations of a graph
        while ((oper = TF_GraphNextOperation(this->graph.get(), &pos)) != nullptr) {
            result.emplace_back(TF_OperationName(oper));
        }
        return result;
    }

    inline std::vector<int64_t> model::get_operation_shape(const std::string& operation) const {
        // Get operation by the name
        TF_Output out_op;
        out_op.oper = TF_GraphOperationByName(this->graph.get(), operation.c_str());
        out_op.index = 0;

        std::vector<int64_t> shape;

        // Operation does not exist
        if (!out_op.oper)
            throw std::runtime_error("No operation named \"" + operation + "\" exists");

        if (operation == "NoOp")
             throw std::runtime_error("NoOp doesn't have a shape");

        // DIMENSIONS

        // Get number of dimensions
        int n_dims = TF_GraphGetTensorNumDims(this->graph.get(), out_op, context::get_status());

        // If is not a scalar
        if (n_dims > 0) {
            // Get dimensions
            auto* dims = new int64_t[n_dims];
            TF_GraphGetTensorShape(this->graph.get(), out_op, dims, n_dims, context::get_status());

            // Check error on Model Status
            status_check(context::get_status());

            shape = std::vector<int64_t>(dims, dims + n_dims);

            delete[] dims;
        }

        return shape;
    }

    inline std::tuple<std::string, int> parse_name(const std::string& name) {
        auto idx = name.find(':');
        return (idx == std::string::npos ? std::make_tuple(name, 0) : std::make_tuple(name.substr(0, idx), std::stoi(name.substr(idx + 1))));
    }

    inline std::vector<tensor> model::operator()(std::vector<std::tuple<std::string, tensor>> inputs, std::vector<std::string> outputs) {

        std::vector<TF_Output> inp_ops(inputs.size());
        std::vector<TF_Tensor*> inp_val(inputs.size(), nullptr);

        for (decltype(inputs.size()) i=0; i<inputs.size(); i++) {

            // Operations
            const auto[op_name, op_idx] = parse_name(std::get<0>(inputs[i]));
            inp_ops[i].oper = TF_GraphOperationByName(this->graph.get(), op_name.c_str());
            inp_ops[i].index = op_idx;

            if (!inp_ops[i].oper)
                throw std::runtime_error("No operation named \"" + op_name + "\" exists");

            // Values
            inp_val[i] = std::get<1>(inputs[i]).get_tensor().get();
        }

        std::vector<TF_Output> out_ops(outputs.size());
        auto out_val = std::make_unique<TF_Tensor*[]>(outputs.size());
        for (decltype(outputs.size()) i=0; i<outputs.size(); i++) {

            const auto[op_name, op_idx] = parse_name(outputs[i]);
            out_ops[i].oper = TF_GraphOperationByName(this->graph.get(), op_name.c_str());
            out_ops[i].index = op_idx;

            if (!out_ops[i].oper)
                throw std::runtime_error("No operation named \"" + op_name + "\" exists");

        }

        TF_SessionRun(this->session.get(), NULL,
                inp_ops.data(), inp_val.data(), static_cast<int>(inputs.size()),
                out_ops.data(), out_val.get(), static_cast<int>(outputs.size()),
                NULL, 0,NULL , context::get_status());
        status_check(context::get_status());

        std::vector<tensor> result;
        result.reserve(outputs.size());
        for (decltype(outputs.size()) i=0; i<outputs.size(); i++) {
            result.emplace_back(tensor(out_val[i]));
        }

        return result;
    }

    inline tensor model::operator()(const tensor& input) {
        return (*this)({{"serving_default_input_1", input}}, {"StatefulPartitionedCall"})[0];
    }


    inline TF_Buffer * model::readGraph(const std::string& filename) {
        std::ifstream file (filename, std::ios::binary | std::ios::ate);

        // Error opening the file
        if (!file.is_open()) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return nullptr;
        }

        // Cursor is at the end to get size
        auto size = file.tellg();
        // Move cursor to the beginning
        file.seekg (0, std::ios::beg);

        // Read
        auto data = std::make_unique<char[]>(size);
        file.seekg (0, std::ios::beg);
        file.read (data.get(), size);

        // Error reading the file
        if (!file) {
            std::cerr << "Unable to read the full file: " << filename << std::endl;
            return nullptr;
        }

        // Create tensorflow buffer from read data
        TF_Buffer* buffer = TF_NewBufferFromString(data.get(), size);

        // Close file and remove data
        file.close();

        return buffer;
    }

}

#endif //CPPFLOW2_MODEL_H
