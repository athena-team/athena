//
// Created by serizba on 27/6/20.
//

#ifndef CPPFLOW2_TENSOR_H
#define CPPFLOW2_TENSOR_H

#include <memory>
#include <vector>
#include <cstring>
#include <string>
#include <tensorflow/c/tf_tensor.h>
#include <tensorflow/c/eager/c_api.h>

#include "context.h"
#include "datatype.h"

namespace cppflow {

    /**
     * @class tensor
     * @brief A TensorFlow eager tensor wrapper
     *
     */
    class tensor {
    public:
        tensor()= default;

        /**
        * Creates a tensor with the given values and specified shape
        * @tparam T A type that can be convertible into a tensor
        * @param values The values to be converted (in a flattened version)
        * @param shape The shape of the converted tensor
        */
        template<typename T>
        tensor(const std::vector<T>& values, const std::vector<int64_t>& shape);

        /**
        * Creates a flat tensor with the given values
        * @tparam T A type that can be convertible into a tensor
        * @param values The values to be converted
        */
        template<typename T>
        tensor(const std::initializer_list<T>& values);

        /**
         * Creates a tensor with the given value
         * @tparam T A type that can be convertible into a tensor
         * @param value The value to be converted
         */
        template<typename T>
        tensor(const T& value);

        /**
         * @return Shape of the tensor
         */
        tensor shape() const;

        /**
         * @param on_memory If false, the function will return the name of the device that produced the tensor.
         * If true, the function will return the name of the device in whose memory the tensor resides
         * @return Returns the name of the device of the tensor
         */
        std::string device(bool on_memory=false) const;


        /**
         * @return The tensor datatype
         */
        datatype dtype() const;

        /**
         * Converts the tensor into a C++ vector
         * @tparam T The c++ type (must be equivalent to the tensor type)
         * @return A vector representing the flat tensor
         */
        template<typename T>
        std::vector<T> get_data() const;


        ~tensor() = default;
        tensor(const tensor &tensor) = default;
        tensor(tensor &&tensor) = default;
        tensor &operator=(const tensor &other) = default;
        tensor &operator=(tensor &&other) = default;

        explicit tensor(TFE_TensorHandle* handle);
        explicit tensor(TF_Tensor* t);

        // NOTE: Usually, one should not call get_eager_handle() or get_tensor() below.
        //       They are designed for implementation details in cppflow.
        //       If you are calling them directly, it is likely that you are using some
        //       tenforflow APIs not supported in cppflow.
        
        // Additional NOTE: TF_Tensor is an immutable tensor inside tensorflow.
        // TFE_TensorHandle is a TF_Tensor and the associated device, plus some data cache
        
        // TODO: Need to determine if we can mark the return value or *this as const
        std::shared_ptr<TFE_TensorHandle> get_eager_handle() const { return tfe_handle; }

        // Get the TF_Tensor data from the eager handle
        // Call `get_data<T>()` instead if possible
        // NOTE: Changes to the returned TF_Tensor may not be reflected in the actual device memory!
        //       Do *NOT* modify the returned TF_Tensor!
        //       See comments of `tf_tensor` for more details.
        std::shared_ptr<TF_Tensor> get_tensor() const;
        
        // DO NOT directly access this member, call get_eager_handle() instead
        // TODO: This is kept as public to be compatible with existing code and should be mark as private
        std::shared_ptr<TFE_TensorHandle> tfe_handle;

    private:
        // This member serves as a local cache of the data in tfe_handle.
        // It refers to `local_mirrors_` if on device, or `data_` if on host CPU.
        // Changes to this variable may not be reflected in the actual device memory,
        // e.g. on GPUs or on remote nodes.
        // Access it via get_tensor() if not in constructor
        mutable std::shared_ptr<TF_Tensor> tf_tensor;

        tensor(enum TF_DataType type, const void* data, size_t len, const std::vector<int64_t>& shape);
    };
}


/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/


namespace cppflow {

    inline tensor::tensor(enum TF_DataType type, const void *data, size_t len, const std::vector<int64_t> &shape) {
        this->tf_tensor = {TF_AllocateTensor(type, shape.data(), static_cast<int>(shape.size()), len), TF_DeleteTensor};
        memcpy(TF_TensorData(this->tf_tensor.get()), data, TF_TensorByteSize(this->tf_tensor.get()));
        this->tfe_handle = {TFE_NewTensorHandle(this->tf_tensor.get(), context::get_status()), TFE_DeleteTensorHandle};
        status_check(context::get_status());
    }

    template<typename T>
    tensor::tensor(const std::vector<T>& values, const std::vector<int64_t>& shape) :
        tensor(deduce_tf_type<T>(), values.data(), values.size() * sizeof(T), shape) {}

    template<typename T>
    tensor::tensor(const std::initializer_list<T>& values) :
            tensor(std::vector<T>(values), {(int64_t) values.size()}) {}

    template<typename T>
    tensor::tensor(const T& value) :
            tensor(std::vector<T>({value}), {}) {}

#ifdef TENSORFLOW_C_TF_TSTRING_H_
    // For future version TensorFlow 2.4
    template<>
    inline tensor::tensor(const std::string& value) {
        TF_TString tstr[1];
        TF_TString_Init(&tstr[0]);
        TF_TString_Copy(&tstr[0], value.c_str(), value.size());

        *this = tensor(static_cast<enum TF_DataType>(TF_STRING), (void *) tstr, sizeof(tstr), {});
    }
#else
    template<>
    inline tensor::tensor(const std::string& value) {
        size_t size = 8 + TF_StringEncodedSize(value.length());
        char* data = new char[value.size() + 8];
        for (int i=0; i<8; i++) {data[i]=0;}
        TF_StringEncode(value.c_str(), value.size(), data + 8, size - 8, context::get_status());
        status_check(context::get_status());

        *this = tensor(static_cast<enum TF_DataType>(TF_STRING), (void *) data, size, {});
        delete [] data;
    }
#endif // TENSORFLOW_C_TF_TSTRING_H_

    inline tensor::tensor(TFE_TensorHandle* handle) {
            this->tfe_handle = {handle, TFE_DeleteTensorHandle};
    }

    inline tensor::tensor(TF_Tensor* t) {
        this->tf_tensor = {t, TF_DeleteTensor};
        this->tfe_handle = {TFE_NewTensorHandle(this->tf_tensor.get(), context::get_status()), TFE_DeleteTensorHandle};
        status_check(context::get_status());
    }

    inline tensor tensor::shape() const {
        auto op = TFE_NewOp(context::get_context(), "Shape", context::get_status());
        status_check(context::get_status());

        TFE_OpAddInput(op, this->tfe_handle.get(), context::get_status());
        status_check(context::get_status());

        // Output type should be int64_t
        TFE_OpSetAttrType(op, "out_type", cppflow::datatype::TF_INT64);

        // EXECUTE
        int n = 1;
        TFE_TensorHandle* res[1] = { nullptr };
        TFE_Execute(op, res, &n, context::get_status());
        status_check(context::get_status());
        TFE_DeleteOp(op);

        return tensor(res[0]);
    }

    inline std::string tensor::device(bool on_memory) const {
        std::string res;
        if (on_memory)
            res = TFE_TensorHandleBackingDeviceName(this->tfe_handle.get(), context::get_status());
        else
            res = std::string(TFE_TensorHandleDeviceName(this->tfe_handle.get(), context::get_status()));

        status_check(context::get_status());
        return res;
    }

    template<typename T>
    std::vector<T> tensor::get_data() const {

        // Check if asked datatype and tensor datatype match
        if (this->dtype() != deduce_tf_type<T>()) {
            auto type1 = cppflow::to_string(deduce_tf_type<T>());
            auto type2 = cppflow::to_string(this->dtype());
            auto error = "Datatype in function get_data (" + type1 + ") does not match tensor datatype (" + type2 + ")";
            throw std::runtime_error(error);
        }


        auto res_tensor = get_tensor();

        // Check tensor data is not empty
        auto raw_data = TF_TensorData(res_tensor.get());
        //this->error_check(raw_data != nullptr, "Tensor data is empty");

        size_t size = TF_TensorByteSize(res_tensor.get()) / TF_DataTypeSize(TF_TensorType(res_tensor.get()));

        // Convert to correct type
        const auto T_data = static_cast<T*>(raw_data);
        std::vector<T> r(T_data, T_data + size);

        return r;
    }

    inline datatype tensor::dtype() const {
        return TFE_TensorHandleDataType(this->tfe_handle.get());
    }
    
    // NOTE: Changes to the returned TF_Tensor are not reflected in the actual device memory!
    inline std::shared_ptr<TF_Tensor> tensor::get_tensor() const {
        if(!tf_tensor) {
            tf_tensor = {TFE_TensorHandleResolve(tfe_handle.get(), context::get_status()), TF_DeleteTensor};
            status_check(context::get_status());
        }
        return tf_tensor;
    }
}

#endif //CPPFLOW2_TENSOR_H
