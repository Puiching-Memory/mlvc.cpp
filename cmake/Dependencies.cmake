# Project and third-party dependencies are resolved in one place.  Backend
# subdirectories consume the resulting imported targets and *_ROOT variables;
# they do not perform global dependency discovery themselves.

set(MLVC_UPSTREAM_DIR "${PROJECT_SOURCE_DIR}/third_party/mlvc")
set(NLOHMANN_JSON_DIR "${PROJECT_SOURCE_DIR}/third_party/nlohmann_json")
set(MLVC_PROFILE_FILE "${PROJECT_SOURCE_DIR}/models/profiles/profiles.json")

if(NOT EXISTS "${MLVC_UPSTREAM_DIR}/packages/msrtc_rans/CMakeLists.txt" OR
   NOT EXISTS "${NLOHMANN_JSON_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Source submodules are missing. Run: "
        "git submodule update --init --recursive --depth 1")
endif()
if(NOT EXISTS "${MLVC_PROFILE_FILE}")
    message(FATAL_ERROR "Model profile registry is missing: ${MLVC_PROFILE_FILE}")
endif()

set(BUILD_PYTHON_MODULE OFF CACHE BOOL
    "Build msrtc_rans Python module" FORCE)
set(INSTALL_ONLY_PYTHON_MODULE ON CACHE BOOL
    "Do not install the internal msrtc_rans static archive" FORCE)
add_subdirectory("${MLVC_UPSTREAM_DIR}/packages/msrtc_rans"
                 "${PROJECT_BINARY_DIR}/_deps/msrtc_rans" EXCLUDE_FROM_ALL)

set(JSON_BuildTests OFF CACHE INTERNAL "Disable nlohmann/json tests")
set(JSON_Install OFF CACHE INTERNAL "Do not install nlohmann/json")
add_subdirectory("${NLOHMANN_JSON_DIR}"
                 "${PROJECT_BINARY_DIR}/_deps/nlohmann_json" EXCLUDE_FROM_ALL)

if(MLVC_SELECTED_BACKEND STREQUAL "onnxruntime")
    set(ONNXRUNTIME_ROOT "" CACHE PATH
        "Root of the ONNX Runtime CUDA package")
    if(NOT ONNXRUNTIME_ROOT)
        message(FATAL_ERROR
            "Set ONNXRUNTIME_ROOT or run tools/bootstrap.sh --backend onnxruntime")
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
    message(STATUS "Backend onnxruntime: ${ONNXRUNTIME_LIBRARY}")

elseif(MLVC_SELECTED_BACKEND STREQUAL "libtorch")
    set(LIBTORCH_ROOT "" CACHE PATH "Root of the libtorch cu130 package")
    if(NOT LIBTORCH_ROOT)
        message(FATAL_ERROR
            "Set LIBTORCH_ROOT or run tools/bootstrap.sh --backend libtorch")
    endif()
    set(CAFFE2_USE_CUDNN ON)
    set(CAFFE2_USE_CUSPARSELT ON)
    set(CUDNN_ROOT "${LIBTORCH_ROOT}" CACHE PATH "" FORCE)
    set(CUDNN_INCLUDE_DIR "${LIBTORCH_ROOT}/include" CACHE PATH "" FORCE)
    set(CUDNN_LIBRARY "${LIBTORCH_ROOT}/lib" CACHE PATH "" FORCE)
    set(CUSPARSELT_ROOT "${LIBTORCH_ROOT}" CACHE PATH "" FORCE)
    set(CUSPARSELT_INCLUDE_DIR "${LIBTORCH_ROOT}/include" CACHE PATH "" FORCE)
    set(CUSPARSELT_LIBRARY "${LIBTORCH_ROOT}/lib" CACHE PATH "" FORCE)
    find_package(Torch REQUIRED PATHS "${LIBTORCH_ROOT}" NO_DEFAULT_PATH)
    if(NOT Torch_VERSION VERSION_EQUAL "2.13.0")
        message(FATAL_ERROR "libtorch 2.13.0 is required, found ${Torch_VERSION}")
    endif()
    find_library(LIBTORCH_CUDA_LIBRARY torch_cuda
        PATHS "${LIBTORCH_ROOT}/lib" NO_DEFAULT_PATH)
    if(NOT LIBTORCH_CUDA_LIBRARY)
        message(FATAL_ERROR "LIBTORCH_ROOT is not the cu130 package")
    endif()
    if(NOT CUDNN_FOUND OR NOT CUSPARSELT_FOUND)
        message(FATAL_ERROR
            "libtorch requires the pinned cuDNN and cuSPARSELt SDKs")
    endif()
    message(STATUS "Backend libtorch: ${LIBTORCH_ROOT}")

elseif(MLVC_SELECTED_BACKEND STREQUAL "tensorrt")
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
    message(STATUS "Backend tensorrt 11.2: ${NVINFER_LIBRARY}")

elseif(MLVC_SELECTED_BACKEND STREQUAL "driver_cubin")
    find_package(CUDAToolkit 13.3 REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    enable_language(ASM)
endif()
