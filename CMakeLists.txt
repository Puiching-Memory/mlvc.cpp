cmake_minimum_required(VERSION 3.23)
project(mlvc_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MLVC_DEMO_BUILD_TESTS "Build tests" OFF)

# Inference backends. libtorch >= 2.13 requires the C++20 toolchain set above.
option(MLVC_WITH_ONNXRUNTIME "ONNX Runtime backend" ON)
option(MLVC_WITH_LIBTORCH "libtorch (TorchScript) backend" OFF)
option(MLVC_WITH_TENSORRT "TensorRT backend" OFF)

# ---------------------------------------------------------------------------
# Third party: msrtc_rans (vendored static library)
# ---------------------------------------------------------------------------
add_subdirectory(third_party/msrtc_rans)

# ---------------------------------------------------------------------------
# nlohmann/json (fetched at configure time)
# ---------------------------------------------------------------------------
# -DMLVC_DEPS_NO_PROXY=ON: fetch dependencies without the system network proxy.
# Unsets the proxy env vars inherited by git/curl. Note: a proxy set via
# `git config http.proxy` is NOT covered; clear it with git config if needed.
option(MLVC_DEPS_NO_PROXY "Fetch dependencies bypassing the system proxy" OFF)
if(MLVC_DEPS_NO_PROXY)
    foreach(proxy_var http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY)
        set(ENV{${proxy_var}} "")
    endforeach()
endif()

include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# ---------------------------------------------------------------------------
# Backend SDKs
# ---------------------------------------------------------------------------
set(MLVC_BACKEND_LIBS "")
set(MLVC_BACKEND_INCLUDES "")
set(MLVC_BACKEND_DEFINES "")

# ONNX Runtime: point ONNXRUNTIME_ROOT at an unpacked package, or set it via
# -DONNXRUNTIME_ROOT=... / the ONNXRUNTIME_ROOT environment variable.
if(MLVC_WITH_ONNXRUNTIME)
    set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of the ONNX Runtime package")
    if(NOT ONNXRUNTIME_ROOT)
        set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
    endif()
    if(NOT ONNXRUNTIME_ROOT)
        # Auto-discover a package fetched by scripts/fetch_onnxruntime.sh
        file(GLOB _ort_candidates LIST_DIRECTORIES true
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/onnxruntime-*")
        if(_ort_candidates)
            list(GET _ort_candidates 0 ONNXRUNTIME_ROOT)
            message(STATUS "Auto-discovered ONNX Runtime: ${ONNXRUNTIME_ROOT}")
        endif()
    endif()
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)
    list(APPEND MLVC_BACKEND_LIBS ${ONNXRUNTIME_LIBRARY})
    list(APPEND MLVC_BACKEND_INCLUDES ${ONNXRUNTIME_INCLUDE_DIR})
    list(APPEND MLVC_BACKEND_DEFINES MLVC_WITH_ONNXRUNTIME)
    message(STATUS "Backend onnxruntime: ${ONNXRUNTIME_LIBRARY}")
endif()

# libtorch: -DLIBTORCH_ROOT=<unpacked libtorch> (scripts/fetch_libtorch.sh).
if(MLVC_WITH_LIBTORCH)
    set(LIBTORCH_ROOT "" CACHE PATH "Root of the libtorch package")
    if(NOT LIBTORCH_ROOT)
        set(LIBTORCH_ROOT "$ENV{LIBTORCH_ROOT}")
    endif()
    if(NOT LIBTORCH_ROOT AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch/libtorch/share/cmake/Torch/TorchConfig.cmake")
        # Auto-discover a package fetched by scripts/fetch_libtorch.sh
        set(LIBTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch/libtorch")
        message(STATUS "Auto-discovered libtorch: ${LIBTORCH_ROOT}")
    endif()
    find_package(Torch REQUIRED
        PATHS "${LIBTORCH_ROOT}" NO_DEFAULT_PATH)
    list(APPEND MLVC_BACKEND_LIBS ${TORCH_LIBRARIES})
    list(APPEND MLVC_BACKEND_INCLUDES ${TORCH_INCLUDE_DIRS})
    list(APPEND MLVC_BACKEND_DEFINES MLVC_WITH_LIBTORCH)
    message(STATUS "Backend libtorch: ${TORCH_LIBRARIES}")
endif()

# TensorRT: -DTENSORRT_ROOT=<unpacked TensorRT tar>; also needs CUDA runtime.
if(MLVC_WITH_TENSORRT)
    set(TENSORRT_ROOT "" CACHE PATH "Root of the TensorRT package")
    if(NOT TENSORRT_ROOT)
        set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
    endif()
    if(NOT TENSORRT_ROOT)
        # Auto-discover a package extracted by scripts/fetch_tensorrt.sh
        file(GLOB _trt_candidates LIST_DIRECTORIES true
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt/TensorRT-*")
        if(_trt_candidates)
            list(GET _trt_candidates 0 TENSORRT_ROOT)
            message(STATUS "Auto-discovered TensorRT: ${TENSORRT_ROOT}")
        endif()
    endif()
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS "${TENSORRT_ROOT}/include" REQUIRED)
    find_library(NVINFER_LIBRARY nvinfer
        HINTS "${TENSORRT_ROOT}/lib" REQUIRED)
    find_library(NVONNXPARSER_LIBRARY nvonnxparser
        HINTS "${TENSORRT_ROOT}/lib" REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND MLVC_BACKEND_LIBS
        ${NVINFER_LIBRARY} ${NVONNXPARSER_LIBRARY} CUDA::cudart)
    list(APPEND MLVC_BACKEND_INCLUDES
        ${TENSORRT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    list(APPEND MLVC_BACKEND_DEFINES MLVC_WITH_TENSORRT)
    message(STATUS "Backend tensorrt: ${NVINFER_LIBRARY}")
endif()

set(MLVC_DEMO_SOURCES
    src/main.cpp
    src/backends/backend.cpp
    src/backends/onnxruntime_backend.cpp
    src/backends/libtorch_backend.cpp
    src/backends/tensorrt_backend.cpp
)

add_executable(mlvc_demo ${MLVC_DEMO_SOURCES})

target_include_directories(mlvc_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${MLVC_BACKEND_INCLUDES}
)

target_compile_definitions(mlvc_demo PRIVATE ${MLVC_BACKEND_DEFINES})

target_link_libraries(mlvc_demo PRIVATE
    msrtc_rans
    nlohmann_json::nlohmann_json
    ${MLVC_BACKEND_LIBS}
)

# Keep the binary relocatable: look for libonnxruntime next to the executable
set_target_properties(mlvc_demo PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
)

install(TARGETS mlvc_demo RUNTIME DESTINATION bin)
