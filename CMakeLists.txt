cmake_minimum_required(VERSION 3.23)
project(mlvc_cpp VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "mlvc.cpp supports Linux x86_64 NVIDIA GPU releases only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MLVC_ENABLE_IPO "Enable interprocedural optimization in Release builds" ON)

# A release contains exactly one inference backend.  Keeping the variants
# separate avoids loading mutually incompatible CUDA/cuDNN copies and makes
# the dependency boundary of each package explicit.
set(MLVC_BACKEND "onnxruntime" CACHE STRING
    "Inference backend: onnxruntime, libtorch, or tensorrt")
set_property(CACHE MLVC_BACKEND PROPERTY STRINGS onnxruntime libtorch tensorrt)

set(MLVC_WITH_ONNXRUNTIME OFF)
set(MLVC_WITH_LIBTORCH OFF)
set(MLVC_WITH_TENSORRT OFF)
if(MLVC_BACKEND STREQUAL "onnxruntime")
    set(MLVC_WITH_ONNXRUNTIME ON)
elseif(MLVC_BACKEND STREQUAL "libtorch")
    set(MLVC_WITH_LIBTORCH ON)
elseif(MLVC_BACKEND STREQUAL "tensorrt")
    set(MLVC_WITH_TENSORRT ON)
else()
    message(FATAL_ERROR
        "MLVC_BACKEND must be one of: onnxruntime, libtorch, tensorrt")
endif()

message(STATUS "MLVC release backend: ${MLVC_BACKEND}")

# ---------------------------------------------------------------------------
# Source dependencies (shallow Git submodules)
# ---------------------------------------------------------------------------
set(MLVC_UPSTREAM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mlvc")
set(NLOHMANN_JSON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann_json")
if(NOT EXISTS "${MLVC_UPSTREAM_DIR}/packages/msrtc_rans/CMakeLists.txt" OR
   NOT EXISTS "${NLOHMANN_JSON_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Source submodules are missing. Run: "
        "git submodule update --init --recursive --depth 1")
endif()

# The upstream msrtc_rans project also builds a Python extension by default;
# this runtime only needs its static C++ target and does not install it.
set(BUILD_PYTHON_MODULE OFF CACHE BOOL "Build msrtc_rans Python module" FORCE)
set(INSTALL_ONLY_PYTHON_MODULE ON CACHE BOOL
    "Do not install the internal msrtc_rans static archive" FORCE)
add_subdirectory("${MLVC_UPSTREAM_DIR}/packages/msrtc_rans"
                 "${CMAKE_CURRENT_BINARY_DIR}/_deps/msrtc_rans" EXCLUDE_FROM_ALL)

set(JSON_BuildTests OFF CACHE INTERNAL "Disable nlohmann/json tests")
set(JSON_Install OFF CACHE INTERNAL "Do not install nlohmann/json")
add_subdirectory("${NLOHMANN_JSON_DIR}"
                 "${CMAKE_CURRENT_BINARY_DIR}/_deps/nlohmann_json" EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Backend SDKs
# ---------------------------------------------------------------------------
set(MLVC_BACKEND_LIBS "")
set(MLVC_BACKEND_INCLUDES "")

# ONNX Runtime 1.26 CUDA 13 package fetched by scripts/fetch_onnxruntime.sh.
if(MLVC_WITH_ONNXRUNTIME)
    set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of the ONNX Runtime package")
    if(NOT ONNXRUNTIME_ROOT)
        message(FATAL_ERROR "Set ONNXRUNTIME_ROOT using fetch_onnxruntime.sh --print-dir")
    endif()
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    find_library(ONNXRUNTIME_CUDA_PROVIDER_LIBRARY onnxruntime_providers_cuda
        PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
    if(NOT ONNXRUNTIME_CUDA_PROVIDER_LIBRARY)
        message(FATAL_ERROR "ONNXRUNTIME_ROOT is not the CUDA 13 GPU package")
    endif()
    list(APPEND MLVC_BACKEND_LIBS ${ONNXRUNTIME_LIBRARY})
    list(APPEND MLVC_BACKEND_INCLUDES ${ONNXRUNTIME_INCLUDE_DIR})
    message(STATUS "Backend onnxruntime: ${ONNXRUNTIME_LIBRARY}")
endif()

# libtorch 2.13 cu130 package fetched by scripts/fetch_libtorch.sh.
if(MLVC_WITH_LIBTORCH)
    set(LIBTORCH_ROOT "" CACHE PATH "Root of the libtorch package")
    if(NOT LIBTORCH_ROOT)
        message(FATAL_ERROR "Set LIBTORCH_ROOT using fetch_libtorch.sh --print-dir")
    endif()
    set(CAFFE2_USE_CUDNN ON)
    set(CAFFE2_USE_CUSPARSELT ON)
    set(CUDNN_ROOT "${LIBTORCH_ROOT}" CACHE PATH "" FORCE)
    set(CUDNN_INCLUDE_DIR "${LIBTORCH_ROOT}/include" CACHE PATH "" FORCE)
    set(CUDNN_LIBRARY "${LIBTORCH_ROOT}/lib" CACHE PATH "" FORCE)
    set(CUSPARSELT_ROOT "${LIBTORCH_ROOT}" CACHE PATH "" FORCE)
    set(CUSPARSELT_INCLUDE_DIR "${LIBTORCH_ROOT}/include" CACHE PATH "" FORCE)
    set(CUSPARSELT_LIBRARY "${LIBTORCH_ROOT}/lib" CACHE PATH "" FORCE)
    find_package(Torch REQUIRED
        PATHS "${LIBTORCH_ROOT}" NO_DEFAULT_PATH)
    if(NOT Torch_VERSION VERSION_EQUAL "2.13.0")
        message(FATAL_ERROR "libtorch 2.13.0 is required, found ${Torch_VERSION}")
    endif()
    find_library(LIBTORCH_CUDA_LIBRARY torch_cuda
        PATHS "${LIBTORCH_ROOT}/lib" NO_DEFAULT_PATH)
    if(NOT LIBTORCH_CUDA_LIBRARY)
        message(FATAL_ERROR "LIBTORCH_ROOT is not the cu130 package")
    endif()
    if(NOT CUDNN_FOUND OR NOT CUSPARSELT_FOUND)
        message(FATAL_ERROR "libtorch requires the pinned cuDNN and cuSPARSELt SDKs")
    endif()
    list(APPEND MLVC_BACKEND_LIBS ${TORCH_LIBRARIES})
    list(APPEND MLVC_BACKEND_INCLUDES ${TORCH_INCLUDE_DIRS})
    message(STATUS "Backend libtorch: ${TORCH_LIBRARIES}")
endif()

# TensorRT 11.2 is installed system-wide from NVIDIA's CUDA apt repository.
if(MLVC_WITH_TENSORRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h REQUIRED)
    file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h"
        _trt_major_line REGEX "^#define TRT_MAJOR_ENTERPRISE +[0-9]+")
    file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h"
        _trt_minor_line REGEX "^#define TRT_MINOR_ENTERPRISE +[0-9]+")
    string(REGEX MATCH "[0-9]+" TENSORRT_MAJOR_VERSION "${_trt_major_line}")
    string(REGEX MATCH "[0-9]+" TENSORRT_MINOR_VERSION "${_trt_minor_line}")
    if(NOT TENSORRT_MAJOR_VERSION EQUAL 11 OR
       NOT TENSORRT_MINOR_VERSION EQUAL 2)
        message(FATAL_ERROR "TensorRT 11.2 is required")
    endif()
    find_library(NVINFER_LIBRARY nvinfer REQUIRED)
    find_library(NVONNXPARSER_LIBRARY nvonnxparser REQUIRED)
    get_filename_component(TENSORRT_LIBRARY_DIR "${NVINFER_LIBRARY}" DIRECTORY)
    find_package(CUDAToolkit 13.3 REQUIRED)
    list(APPEND MLVC_BACKEND_LIBS
        ${NVINFER_LIBRARY} ${NVONNXPARSER_LIBRARY} CUDA::cudart)
    list(APPEND MLVC_BACKEND_INCLUDES
        ${TENSORRT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "Backend tensorrt 11.2: ${NVINFER_LIBRARY}")
endif()

set(MLVC_DEMO_SOURCES
    src/main.cpp
)
if(MLVC_WITH_ONNXRUNTIME)
    list(APPEND MLVC_DEMO_SOURCES src/backends/onnxruntime_backend.cpp)
elseif(MLVC_WITH_LIBTORCH)
    list(APPEND MLVC_DEMO_SOURCES src/backends/libtorch_backend.cpp)
else()
    list(APPEND MLVC_DEMO_SOURCES src/backends/tensorrt_backend.cpp)
endif()

add_executable(mlvc_demo ${MLVC_DEMO_SOURCES})

target_include_directories(mlvc_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${MLVC_BACKEND_INCLUDES}
)

target_link_libraries(mlvc_demo PRIVATE
    msrtc_rans
    nlohmann_json::nlohmann_json
    ${MLVC_BACKEND_LIBS}
)
target_link_options(mlvc_demo PRIVATE "LINKER:--disable-new-dtags")
if(MLVC_WITH_LIBTORCH)
    target_link_options(mlvc_demo PRIVATE
        "LINKER:-rpath-link,${LIBTORCH_ROOT}/lib")
endif()

if(MLVC_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(LANGUAGES CXX
        RESULT MLVC_IPO_SUPPORTED OUTPUT MLVC_IPO_ERROR)
    if(MLVC_IPO_SUPPORTED)
        set_property(TARGET mlvc_demo msrtc_rans PROPERTY
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    else()
        message(WARNING "IPO is unavailable: ${MLVC_IPO_ERROR}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Portable install layout: the binary goes to bin/ and every backend's
# runtime libraries to lib/, found via $ORIGIN/../lib (no LD_LIBRARY_PATH).
# ---------------------------------------------------------------------------
set(MLVC_LIB_DEST "lib")
set_target_properties(mlvc_demo PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN/../lib"
)

install(TARGETS mlvc_demo RUNTIME DESTINATION bin)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
    DESTINATION "share/licenses/mlvc_cpp")

# Install each selected backend's shared libraries. CUDA 13.3 remains the one
# system prerequisite; TensorRT only needs its core runtime and ONNX parser.
if(MLVC_WITH_ONNXRUNTIME)
    file(GLOB _onnxruntime_runtime_libraries
        "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*"
        "${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_shared.so*"
        "${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_cuda.so*"
        "${ONNXRUNTIME_ROOT}/lib/libcudnn*.so*")
    install(FILES ${_onnxruntime_runtime_libraries}
        DESTINATION "${MLVC_LIB_DEST}")
endif()
if(MLVC_WITH_LIBTORCH)
    file(GLOB _libtorch_runtime_libraries "${LIBTORCH_ROOT}/lib/*.so*")
    install(FILES ${_libtorch_runtime_libraries}
        DESTINATION "${MLVC_LIB_DEST}")
endif()
if(MLVC_WITH_TENSORRT)
    file(GLOB _tensorrt_runtime_libraries
        "${TENSORRT_LIBRARY_DIR}/libnvinfer.so*"
        "${TENSORRT_LIBRARY_DIR}/libnvinfer_builder_resource_*.so*"
        "${TENSORRT_LIBRARY_DIR}/libnvonnxparser.so*")
    install(FILES ${_tensorrt_runtime_libraries}
        DESTINATION "${MLVC_LIB_DEST}")
endif()

set(_backend_license_root "")
if(MLVC_WITH_ONNXRUNTIME)
    set(_backend_license_root "${ONNXRUNTIME_ROOT}")
elseif(MLVC_WITH_LIBTORCH)
    set(_backend_license_root "${LIBTORCH_ROOT}")
endif()
if(_backend_license_root)
    file(GLOB _backend_license_files
        "${_backend_license_root}/LICENSE*"
        "${_backend_license_root}/*Notice*"
        "${_backend_license_root}/*NOTICE*")
    if(_backend_license_files)
        install(FILES ${_backend_license_files}
            DESTINATION "share/licenses/${MLVC_BACKEND}")
    endif()
endif()
if(MLVC_WITH_TENSORRT)
    install(FILES "/usr/share/doc/libnvinfer11/copyright"
        DESTINATION "share/licenses/tensorrt"
        RENAME "libnvinfer11.txt")
    install(FILES "/usr/share/doc/libnvonnxparsers11/copyright"
        DESTINATION "share/licenses/tensorrt"
        RENAME "libnvonnxparsers11.txt")
endif()
