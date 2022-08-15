//
// Created by serizba on 17/9/20.
//

#ifndef EXAMPLE_CPPFLOW_H
#define EXAMPLE_CPPFLOW_H

#include "tensor.h"
#include "model.h"
#include "raw_ops.h"
#include "ops.h"
#include "datatype.h"

#include <tensorflow/c/c_api.h>

namespace cppflow {

    /**
     * Version of TensorFlow and CppFlow
     * @return A string containing the version of TensorFow and CppFlow
     */
    std::string version();

}

/******************************
 *   IMPLEMENTATION DETAILS   *
 ******************************/

namespace cppflow {
    inline std::string version() {
        return "TensorFlow: " + std::string(TF_Version()) + " CppFlow: 2.0.0";
    }
}

#endif //EXAMPLE_CPPFLOW_H
