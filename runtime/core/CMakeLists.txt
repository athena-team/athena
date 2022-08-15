cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(athena VERSION 0.1)
set(CMAKE_VERBOSE_MAKEFILE on)

include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET off)
get_filename_component(third_party "third_party" REALPATH BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(FETCHCONTENT_BASE_DIR ${third_party})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -fPIC")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# third_party: gflags
FetchContent_Declare(gflags
        URL      https://github.com/gflags/gflags/archive/v2.2.1.zip
        URL_HASH SHA256=4e44b69e709c826734dbbbd5208f61888a2faf63f239d73d8ba0011b2dccc97a
        )
FetchContent_MakeAvailable(gflags)
include_directories(${gflags_BINARY_DIR}/include)

# third_party: glog
FetchContent_Declare(glog
        URL      https://github.com/google/glog/archive/v0.4.0.zip
        URL_HASH SHA256=9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc
        )
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})

# third_party: libtensorflow use FetchContent_Declare to download, and
# use find_package to find since libtensorflow is not a standard cmake project
set(TENSORFLOW_VERSION "2.3.0")
set(LIBTENSORFLOW_URL "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-${TENSORFLOW_VERSION}.tar.gz")
set(URL_HASH "SHA256=64b769742e582d4c45cfb751647caf1c9c298240bbd7af40536eb938d3f3e89b")
FetchContent_Declare(libtensorflow
        URL      ${LIBTENSORFLOW_URL}
        URL_HASH ${URL_HASH}
        )
FetchContent_MakeAvailable(libtensorflow)
include_directories(${libtensorflow_SOURCE_DIR}/include)
link_directories(${libtensorflow_SOURCE_DIR}/lib)

# utils
add_library(utils STATIC
        utils/utils.cc
        )

# frontend
add_library(frontend STATIC
        frontend/feature_pipeline.cc
        frontend/recycling_vector.cc
        frontend/fft.cc
        )
target_link_libraries(frontend PUBLIC utils gflags glog)

# decoder
add_library(decoder STATIC
        inference/resource.cc
        inference/interface.cc
        inference/e2e_context.cc
        inference/wfst_beam_search_context.cc
        inference/prefix_beam_search_context.cc
        decoder/ctc_decodable.cc
        decoder/ctc_faster_decoder.cc
        decoder/ctc_prefix_beam_search.cc
        fst/graph_io.cc
        )
target_link_libraries(decoder PUBLIC tensorflow frontend)

add_executable(decoder_main bin/decoder_main.cc)
target_link_libraries(decoder_main PUBLIC decoder)