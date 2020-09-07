cmake_minimum_required (VERSION 2.8)
cmake_policy(SET CMP0015 NEW)

# project name
project(RunGraph)

# include header file search directires
include_directories("./dependencies/tensorflow")
include_directories("./dependencies/tensorflow/bazel-genfiles")
include_directories("./dependencies/tensorflow/tensorflow/contrib/makefile/gen/protobuf/include")
include_directories("./dependencies/tensorflow/tensorflow/contrib/makefile/downloads/absl")
include_directories("./dependencies/eigen")
include_directories("./include")

# add directories where the linker should search for libraries
link_directories("./dependencies/tensorflow/bazel-bin/tensorflow/")

math(EXPR BITNESS "8*${CMAKE_SIZEOF_VOID_P}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g3 -O2 -D_GLIBCXX_USE_NANOSLEEP")

add_library(utils ./include/utils.h ./src/utils.cpp)
add_library(tensor_utils ./include/tensor_utils.h ./src/tensor_utils.cpp)
add_executable(asr ./src/asr.cpp)
add_executable(tts ./src/tts.cpp)

target_link_libraries(
    tensor_utils
        tensorflow_cc
        tensorflow_framework
)

target_link_libraries(
    asr
        tensor_utils
        utils
        tensorflow_cc
        tensorflow_framework
)

target_link_libraries(
    tts
        tensor_utils
        utils
        tensorflow_cc
        tensorflow_framework
)
