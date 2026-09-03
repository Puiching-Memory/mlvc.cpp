cmake_minimum_required(VERSION 3.23)
project(mlvc_cpp VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

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
option(MLVC_DRIVER_CUBIN_ONLY
    "Build the CUDA Runtime-free driver-cubin backend and its probe" OFF)
set(MLVC_TEST_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/model-assets" CACHE PATH
    "Optional local MLVC model/reference assets used by GPU conformance tests")
set(MLVC_EMBEDDED_MODEL_ROOT "${MLVC_TEST_ASSETS_DIR}/models" CACHE PATH
    "Canonical model bundles embedded by the driver-cubin backend")
set(MLVC_TEST_ENGINE_CACHE_DIR "" CACHE PATH
    "Optional TensorRT engine-cache root used by codec conformance tests")

if(MLVC_DRIVER_CUBIN_ONLY)
    set(MLVC_BACKEND "driver_cubin" CACHE STRING "Inference backend" FORCE)
endif()

# A release contains exactly one inference backend.  Keeping the variants
# separate avoids loading mutually incompatible CUDA/cuDNN copies and makes
# the dependency boundary of each package explicit.
set(MLVC_BACKEND "onnxruntime" CACHE STRING
    "Inference backend: onnxruntime, libtorch, tensorrt, or driver_cubin")
set_property(CACHE MLVC_BACKEND PROPERTY STRINGS
    onnxruntime libtorch tensorrt driver_cubin)

set(MLVC_WITH_ONNXRUNTIME OFF)
set(MLVC_WITH_LIBTORCH OFF)
set(MLVC_WITH_TENSORRT OFF)
set(MLVC_WITH_DRIVER_CUBIN OFF)
if(MLVC_BACKEND STREQUAL "onnxruntime")
    set(MLVC_WITH_ONNXRUNTIME ON)
elseif(MLVC_BACKEND STREQUAL "libtorch")
    set(MLVC_WITH_LIBTORCH ON)
elseif(MLVC_BACKEND STREQUAL "tensorrt")
    set(MLVC_WITH_TENSORRT ON)
elseif(MLVC_BACKEND STREQUAL "driver_cubin")
    set(MLVC_WITH_DRIVER_CUBIN ON)
else()
    message(FATAL_ERROR
        "MLVC_BACKEND must be one of: onnxruntime, libtorch, tensorrt, driver_cubin")
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

# Driver-cubin embeds the complete fixed-shape MLVC operator module. The CUDA
# Toolkit supplies headers and the libcuda link stub only; deployed binaries
# do not depend on cudart, cuDNN, TensorRT, ONNX Runtime, or libtorch.
if(MLVC_WITH_DRIVER_CUBIN)
    include(cmake/DriverCubinTargets.cmake)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    enable_language(ASM)
    file(GLOB_RECURSE MLVC_EMBEDDED_MODEL_INPUTS CONFIGURE_DEPENDS
        LIST_DIRECTORIES false "${MLVC_EMBEDDED_MODEL_ROOT}/*")
    if(NOT MLVC_EMBEDDED_MODEL_INPUTS)
        message(FATAL_ERROR
            "No model bundles found below ${MLVC_EMBEDDED_MODEL_ROOT}")
    endif()
    set(MLVC_EMBEDDED_MODELS_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/generated/embedded-models")
    set(MLVC_EMBEDDED_MODELS_BINARY
        "${MLVC_EMBEDDED_MODELS_DIR}/mlvc_driver_models.bin")
    set(MLVC_EMBEDDED_MODELS_INDEX
        "${MLVC_EMBEDDED_MODELS_DIR}/mlvc_embedded_models_index.inc")
    set(MLVC_EMBEDDED_MODELS_ASSEMBLY
        "${MLVC_EMBEDDED_MODELS_DIR}/mlvc_embedded_models.S")
    file(MAKE_DIRECTORY "${MLVC_EMBEDDED_MODELS_DIR}")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedModelData.S.in"
        "${MLVC_EMBEDDED_MODELS_ASSEMBLY}" @ONLY)
    add_custom_command(
        OUTPUT
            "${MLVC_EMBEDDED_MODELS_BINARY}"
            "${MLVC_EMBEDDED_MODELS_INDEX}"
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_embedded_models.py"
            --model-root "${MLVC_EMBEDDED_MODEL_ROOT}"
            --output-data "${MLVC_EMBEDDED_MODELS_BINARY}"
            --output-index "${MLVC_EMBEDDED_MODELS_INDEX}"
        DEPENDS
            ${MLVC_EMBEDDED_MODEL_INPUTS}
            "${CMAKE_CURRENT_SOURCE_DIR}/configs/model_profiles.json"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/assemble_model_package.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_embedded_models.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/package_model_bundles.py"
        COMMENT "Embedding validated driver-cubin model bundles"
        VERBATIM)
    list(APPEND MLVC_BACKEND_LIBS
        mlvc_driver mlvc_driver_kernels nlohmann_json::nlohmann_json)
endif()

if(MLVC_WITH_ONNXRUNTIME)
    set(MLVC_BACKEND_SOURCE src/backends/onnxruntime_backend.cpp)
elseif(MLVC_WITH_LIBTORCH)
    set(MLVC_BACKEND_SOURCE src/backends/libtorch_backend.cpp)
elseif(MLVC_WITH_TENSORRT)
    set(MLVC_BACKEND_SOURCE src/backends/tensorrt_backend.cpp)
else()
    set(MLVC_BACKEND_SOURCE src/backends/driver_cubin_backend.cpp)
endif()

add_library(mlvc_backend STATIC
    ${MLVC_BACKEND_SOURCE}
    src/model_assets.cpp
)
if(MLVC_WITH_DRIVER_CUBIN)
    set_source_files_properties("${MLVC_EMBEDDED_MODELS_INDEX}" PROPERTIES
        GENERATED TRUE
        HEADER_FILE_ONLY TRUE)
    set_source_files_properties("${MLVC_EMBEDDED_MODELS_ASSEMBLY}" PROPERTIES
        OBJECT_DEPENDS "${MLVC_EMBEDDED_MODELS_BINARY}")
    set_source_files_properties(src/model_assets.cpp PROPERTIES
        OBJECT_DEPENDS "${MLVC_EMBEDDED_MODELS_INDEX}")
    target_sources(mlvc_backend PRIVATE
        "${MLVC_EMBEDDED_MODELS_INDEX}"
        "${MLVC_EMBEDDED_MODELS_ASSEMBLY}")
    target_compile_definitions(mlvc_backend PRIVATE MLVC_HAS_EMBEDDED_MODELS=1)
endif()
set_target_properties(mlvc_backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mlvc_backend
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${MLVC_BACKEND_INCLUDES}
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${NLOHMANN_JSON_DIR}/single_include"
)
if(MLVC_WITH_DRIVER_CUBIN)
    target_include_directories(mlvc_backend PRIVATE
        "${MLVC_EMBEDDED_MODELS_DIR}")
endif()
target_link_libraries(mlvc_backend PUBLIC ${MLVC_BACKEND_LIBS})

add_library(mlvc_codec SHARED
    src/bitstream.cpp
    src/c_api.cpp
    src/entropy.cpp
    src/model.cpp
    src/pipeline.cpp
    src/scales.cpp
    src/yuv.cpp
)
target_include_directories(mlvc_codec PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(mlvc_codec PRIVATE
    mlvc_backend
    msrtc_rans
    nlohmann_json::nlohmann_json
)
set_target_properties(mlvc_codec PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME mlvc_codec
    POSITION_INDEPENDENT_CODE ON
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
)

add_executable(mlvc_demo src/main.cpp)
add_executable(mlvc_backend_bench src/backend_bench.cpp)

include(CTest)
if(BUILD_TESTING)
    add_executable(mlvc_codec_tests tests/codec_tests.cpp)
    target_link_libraries(mlvc_codec_tests PRIVATE mlvc_codec)
    add_executable(mlvc_c_api_tests tests/c_api_tests.cpp)
    target_link_libraries(mlvc_c_api_tests PRIVATE mlvc_codec)
    add_test(NAME mlvc_c_api COMMAND mlvc_c_api_tests)
    if(EXISTS "${MLVC_TEST_ASSETS_DIR}/benchmarks/gray-q21/encoder/case.json"
       AND EXISTS "${MLVC_TEST_ASSETS_DIR}/models/mlvc-psnr-v1/canonical/640x368/metadata.json")
        add_test(NAME mlvc_codec_reference
            COMMAND mlvc_codec_tests ${MLVC_TEST_ASSETS_DIR})
    else()
        message(STATUS
            "Portable codec fixtures not found under ${MLVC_TEST_ASSETS_DIR}; "
            "mlvc_codec_reference will not be registered")
    endif()

    set(_mlvc_conformance_script
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_codec_conformance.py")
    if(EXISTS "${MLVC_TEST_ASSETS_DIR}/references/mlvc-psnr-v1/gray-q21-2f/reference.json"
       AND EXISTS "${MLVC_TEST_ASSETS_DIR}/references/mlvc-s-psnr-v1/gray-q21-2f/reference.json")
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        foreach(_mlvc_profile mlvc-psnr-v1 mlvc-s-psnr-v1)
            set(_mlvc_model_dir
                "${MLVC_TEST_ASSETS_DIR}/models/${_mlvc_profile}/canonical/640x368")
            set(_mlvc_reference_dir
                "${MLVC_TEST_ASSETS_DIR}/references/${_mlvc_profile}/gray-q21-2f")
            if(MLVC_WITH_DRIVER_CUBIN)
                set(_mlvc_conformance_args
                    --binary $<TARGET_FILE:mlvc_demo>
                    --model-dir "embedded:${_mlvc_profile}"
                    --bundle-dir "${_mlvc_model_dir}"
                    --reference-dir "${_mlvc_reference_dir}")
            else()
                set(_mlvc_conformance_args
                    --binary $<TARGET_FILE:mlvc_demo>
                    --model-dir "${_mlvc_model_dir}"
                    --reference-dir "${_mlvc_reference_dir}")
            endif()
            if(MLVC_WITH_TENSORRT AND MLVC_TEST_ENGINE_CACHE_DIR)
                list(APPEND _mlvc_conformance_args
                    --engine-cache-dir "${MLVC_TEST_ENGINE_CACHE_DIR}")
            endif()
            add_test(NAME "mlvc_codec_conformance_${_mlvc_profile}"
                COMMAND "${Python3_EXECUTABLE}" "${_mlvc_conformance_script}"
                        ${_mlvc_conformance_args})
            set_tests_properties("mlvc_codec_conformance_${_mlvc_profile}"
                PROPERTIES LABELS "gpu;conformance" TIMEOUT 300)
        endforeach()
        add_custom_target(mlvc_codec_compatibility
            COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L conformance
            DEPENDS mlvc_demo
            USES_TERMINAL
            COMMENT "Checking both MLVC profiles against official Python codec references")
    else()
        message(STATUS
            "Codec conformance references not found under ${MLVC_TEST_ASSETS_DIR}; "
            "only portable codec tests will be registered")
        add_custom_target(mlvc_codec_compatibility
            COMMAND "${CMAKE_COMMAND}" -E echo
                "Missing MLVC and MLVC-S references under ${MLVC_TEST_ASSETS_DIR}"
            COMMAND "${CMAKE_COMMAND}" -E false
            USES_TERMINAL
            COMMENT "Codec compatibility references are required")
    endif()
endif()

target_include_directories(mlvc_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_include_directories(mlvc_backend_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mlvc_demo PRIVATE
    mlvc_codec
)
if(MLVC_WITH_DRIVER_CUBIN)
    target_link_libraries(mlvc_backend_bench PRIVATE
        mlvc_codec
        nlohmann_json::nlohmann_json)
else()
    target_link_libraries(mlvc_backend_bench PRIVATE
        mlvc_backend
        nlohmann_json::nlohmann_json)
endif()

foreach(_mlvc_executable mlvc_demo mlvc_backend_bench)
    target_link_options(${_mlvc_executable} PRIVATE "LINKER:--disable-new-dtags")
endforeach()
if(BUILD_TESTING)
    foreach(_mlvc_test_target mlvc_codec_tests mlvc_c_api_tests)
        target_link_options(${_mlvc_test_target} PRIVATE
            "LINKER:--disable-new-dtags")
    endforeach()
endif()
if(MLVC_WITH_LIBTORCH)
    foreach(_mlvc_executable mlvc_demo mlvc_backend_bench)
        target_link_options(${_mlvc_executable} PRIVATE
            "LINKER:-rpath-link,${LIBTORCH_ROOT}/lib")
    endforeach()
endif()

if(MLVC_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(LANGUAGES CXX
        RESULT MLVC_IPO_SUPPORTED OUTPUT MLVC_IPO_ERROR)
    if(MLVC_IPO_SUPPORTED)
        set_property(TARGET mlvc_backend mlvc_codec mlvc_demo mlvc_backend_bench msrtc_rans PROPERTY
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        if(BUILD_TESTING)
            set_property(TARGET mlvc_codec_tests mlvc_c_api_tests PROPERTY
                INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        endif()
    else()
        message(WARNING "IPO is unavailable: ${MLVC_IPO_ERROR}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Portable install layout: the binary goes to bin/ and every backend's
# runtime libraries to lib/, found via $ORIGIN/../lib (no LD_LIBRARY_PATH).
# ---------------------------------------------------------------------------
set(MLVC_LIB_DEST "${CMAKE_INSTALL_LIBDIR}")
foreach(_mlvc_executable mlvc_demo mlvc_backend_bench)
    set_target_properties(${_mlvc_executable} PROPERTIES
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN/../lib"
    )
endforeach()

install(TARGETS mlvc_codec
    EXPORT mlvc_codecTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(TARGETS mlvc_demo mlvc_backend_bench
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
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

# Export the stable shared-library target for CMake consumers.  The selected
# backend remains an implementation detail of libmlvc_codec.so; consumers do
# not need to link TensorRT or the internal static targets explicitly.
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mlvc_codecConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mlvc_codecConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mlvc_codec"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mlvc_codecConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(EXPORT mlvc_codecTargets
    FILE mlvc_codecTargets.cmake
    NAMESPACE mlvc::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mlvc_codec")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/mlvc_codecConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/mlvc_codecConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mlvc_codec")

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
