//
// Created by serizba on 12/7/20.
//

#ifndef CPPFLOW2_DATATYPE_H
#define CPPFLOW2_DATATYPE_H

#include <type_traits>
#include <string>
#include <typeinfo>
#include <ostream>
#include <stdexcept>

namespace cppflow {

    using datatype = TF_DataType;

    /**
     * @return A string representing dt
     *
     */
    inline std::string to_string(datatype dt) {
        switch (dt) {
            case TF_FLOAT:
                return "TF_FLOAT";
            case TF_DOUBLE:
                return "TF_DOUBLE";
            case TF_INT32:
                return "TF_INT32";
            case TF_UINT8:
                return "TF_UINT8";
            case TF_INT16:
                return "TF_INT16";
            case TF_INT8:
                return "TF_INT8";
            case TF_STRING:
                return "TF_STRING";
            case TF_COMPLEX64:
                return "TF_COMPLEX64";
            case TF_INT64:
                return "TF_INT64";
            case TF_BOOL:
                return "TF_BOOL";
            case TF_QINT8:
                return "TF_QINT8";
            case TF_QUINT8:
                return "TF_QUINT8";
            case TF_QINT32:
                return "TF_QINT32";
            case TF_BFLOAT16:
                return "TF_BFLOAT16";
            case TF_QINT16:
                return "TF_QINT16";
            case TF_QUINT16:
                return "TF_QUINT16";
            case TF_UINT16:
                return "TF_UINT16";
            case TF_COMPLEX128:
                return "TF_COMPLEX128";
            case TF_HALF:
                return "TF_HALF";
            case TF_RESOURCE:
                return "TF_RESOURCE";
            case TF_VARIANT:
                return "TF_VARIANT";
            case TF_UINT32:
                return "TF_UINT32";
            case TF_UINT64:
                return "TF_UINT64";
            default:
                return "DATATYPE_NOT_KNOWN";
        }
    }

    /**
     *
     * @tparam T
     * @return The TensorFlow type of T
     */
    template<typename T>
    TF_DataType deduce_tf_type() {
        if (std::is_same<T, float>::value)
            return TF_FLOAT;
        if (std::is_same<T, double>::value)
            return TF_DOUBLE;
        if (std::is_same<T, int32_t >::value)
            return TF_INT32;
        if (std::is_same<T, uint8_t>::value)
            return TF_UINT8;
        if (std::is_same<T, int16_t>::value)
            return TF_INT16;
        if (std::is_same<T, int8_t>::value)
            return TF_INT8;
        if (std::is_same<T, int64_t>::value)
            return TF_INT64;
        if (std::is_same<T, unsigned char>::value)
            return TF_BOOL;
        if (std::is_same<T, uint16_t>::value)
            return TF_UINT16;
        if (std::is_same<T, uint32_t>::value)
            return TF_UINT32;
        if (std::is_same<T, uint64_t>::value)
            return TF_UINT64;

        // decode with `c++filt --type $output` for gcc
        throw std::runtime_error{"Could not deduce type! type_name: " + std::string(typeid(T).name())};
    }

    /**
     * @return  The stream os after inserting the string representation of dt
     *
     */
    inline std::ostream& operator<<(std::ostream& os, datatype dt) {
        os << to_string(dt);
        return os;
    }

}
#endif //CPPFLOW2_DATATYPE_H
