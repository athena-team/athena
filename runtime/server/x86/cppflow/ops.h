//
// Created by serizba on 31/7/20.
//

#ifndef CPPFLOW2_OPS_H
#define CPPFLOW2_OPS_H


#include "tensor.h"
#include "raw_ops.h"

namespace cppflow {

    /**
     * @name Operators
     */
    //@{

    /**
     * @returns x + y elementwise
     */
    tensor operator+(const tensor& x, const tensor& y);

    /**
     * @returns x - y elementwise
     */
    tensor operator-(const tensor& x, const tensor& y);

    /**
     * @returns x * y elementwise
     */
    tensor operator*(const tensor& x, const tensor& y);

    /**
     * @return x / y elementwise
     */
    tensor operator/(const tensor& x, const tensor& y);

    std::ostream& operator<<(std::ostream& os, const cppflow::tensor& t);

    //@}

    /**
     * @return A string representing t in the form:
     * (tensor: shape=?, data=
     * ?)
     */
    std::string to_string(const tensor& t);
}

/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/

namespace cppflow {

    // Operators

    inline tensor operator+(const tensor& x, const tensor& y) {
        return add(x, y);
    }

    inline tensor operator-(const tensor& x, const tensor& y) {
        return sub(x, y);
    }

    inline tensor operator*(const tensor& x, const tensor& y) {
        return mul(x, y);
    }

    inline tensor operator/(const tensor& x, const tensor& y) {
        return div(x, y);
    }

    inline std::ostream& operator<<(std::ostream& os, const cppflow::tensor& t) {
        std::string res =  to_string(t);
        return os << res;
    }


    inline std::string to_string(const tensor &t) {
        auto res_tensor = string_format({t.shape(), t}, "(tensor: shape=%s, dtype="+ to_string(t.dtype()) + ", data=\n%s)");
        auto res_tensor_h = res_tensor.get_tensor();

#ifdef TENSORFLOW_C_TF_TSTRING_H_
        // For future version TensorFlow 2.4
        //auto *t_str = reinterpret_cast<TF_TString *>(TF_TensorData(res_tensor_h.get()));
        auto *t_str = (TF_TString *)(TF_TensorData(res_tensor_h.get()));
        auto result = std::string(TF_TString_GetDataPointer(t_str), TF_TString_GetSize(t_str));
#else
        const char* dst[1] = {nullptr};
        size_t dst_len[1] = {3};
        TF_StringDecode(static_cast<char*>(TF_TensorData(res_tensor_h.get())) + 8, TF_TensorByteSize(res_tensor_h.get()), dst, dst_len, context::get_status());
        status_check(context::get_status());
        auto result = std::string(dst[0], *dst_len);
#endif // TENSORFLOW_C_TF_TSTRING_H_

        return result;
    }

}

#endif //CPPFLOW2_OPS_H
