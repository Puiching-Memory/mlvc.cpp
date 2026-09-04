cmake_minimum_required(VERSION 3.23)
project(mlvc_cpp VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "mlvc.cpp supports Linux x86_64 NVIDIA GPU releases only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MLVC_ENABLE_IPO "Enable interprocedural optimization in Release builds" ON)
set(MLVC_SELECTED_BACKEND "onnxruntime" CACHE STRING
    "Inference backend: onnxruntime, libtorch, tensorrt, or driver_cubin")
set_property(CACHE MLVC_SELECTED_BACKEND PROPERTY STRINGS
    onnxruntime libtorch tensorrt driver_cubin)
set(MLVC_TEST_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures" CACHE PATH
    "Local MLVC model/reference fixtures used by tests")
set(MLVC_MODEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/models/canonical" CACHE PATH
    "Canonical model bundle root")
set(MLVC_TEST_ENGINE_CACHE_DIR "" CACHE PATH
    "Optional TensorRT engine-cache root used by conformance tests")

set(_mlvc_valid_backends onnxruntime libtorch tensorrt driver_cubin)
if(NOT MLVC_SELECTED_BACKEND IN_LIST _mlvc_valid_backends)
    message(FATAL_ERROR
        "MLVC_SELECTED_BACKEND must be one of: "
        "onnxruntime, libtorch, tensorrt, driver_cubin")
endif()
message(STATUS "MLVC selected backend: ${MLVC_SELECTED_BACKEND}")

include(CheckIPOSupported)
if(MLVC_ENABLE_IPO)
    check_ipo_supported(LANGUAGES CXX
        RESULT MLVC_IPO_SUPPORTED OUTPUT MLVC_IPO_ERROR)
    if(NOT MLVC_IPO_SUPPORTED)
        message(WARNING "IPO is unavailable: ${MLVC_IPO_ERROR}")
    endif()
else()
    set(MLVC_IPO_SUPPORTED OFF)
endif()

function(mlvc_enable_ipo target)
    if(MLVC_ENABLE_IPO AND MLVC_IPO_SUPPORTED AND TARGET "${target}")
        set_property(TARGET "${target}"
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endfunction()

include(cmake/Dependencies.cmake)

add_subdirectory(core)
add_subdirectory(runtime)
add_subdirectory(backends)
add_subdirectory(codec)
add_subdirectory(tools)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
    DESTINATION "share/licenses/mlvc_cpp")

# Export only the stable C ABI. Core, runtime, backend, driver, CUTLASS, and
# AOT headers are implementation details and are intentionally not installed.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/public/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/mlvc_codecConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mlvc_codecConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mlvc_codec")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/mlvc_codecConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
install(EXPORT mlvc_codecTargets
    FILE mlvc_codecTargets.cmake
    NAMESPACE mlvc::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mlvc_codec")
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/mlvc_codecConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/mlvc_codecConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/mlvc_codec")

if(MLVC_SELECTED_BACKEND STREQUAL "onnxruntime")
    file(GLOB _backend_license_files
        "${ONNXRUNTIME_ROOT}/LICENSE*"
        "${ONNXRUNTIME_ROOT}/*Notice*"
        "${ONNXRUNTIME_ROOT}/*NOTICE*")
    if(_backend_license_files)
        install(FILES ${_backend_license_files}
            DESTINATION "share/licenses/onnxruntime")
    endif()
elseif(MLVC_SELECTED_BACKEND STREQUAL "libtorch")
    file(GLOB _backend_license_files
        "${LIBTORCH_ROOT}/LICENSE*"
        "${LIBTORCH_ROOT}/*Notice*"
        "${LIBTORCH_ROOT}/*NOTICE*")
    if(_backend_license_files)
        install(FILES ${_backend_license_files}
            DESTINATION "share/licenses/libtorch")
    endif()
elseif(MLVC_SELECTED_BACKEND STREQUAL "tensorrt")
    if(EXISTS "/usr/share/doc/libnvinfer11/copyright")
        install(FILES "/usr/share/doc/libnvinfer11/copyright"
            DESTINATION "share/licenses/tensorrt"
            RENAME "libnvinfer11.txt")
    endif()
    if(EXISTS "/usr/share/doc/libnvonnxparsers11/copyright")
        install(FILES "/usr/share/doc/libnvonnxparsers11/copyright"
            DESTINATION "share/licenses/tensorrt"
            RENAME "libnvonnxparsers11.txt")
    endif()
endif()
