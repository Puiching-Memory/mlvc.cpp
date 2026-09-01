cmake_minimum_required(VERSION 3.23)
project(mlvc_cpp_demo VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MLVC_DEMO_BUILD_TESTS "Build tests" OFF)

# ---------------------------------------------------------------------------
# Third party: msrtc_rans (vendored static library)
# ---------------------------------------------------------------------------
add_subdirectory(third_party/msrtc_rans)

# ---------------------------------------------------------------------------
# nlohmann/json (fetched at configure time)
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# ---------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------
# Point ONNXRUNTIME_ROOT at an unpacked ONNX Runtime package, or set it via
# -DONNXRUNTIME_ROOT=... / the ONNXRUNTIME_ROOT environment variable.
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of the ONNX Runtime package")
if(NOT ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib" REQUIRED)

set(MLVC_DEMO_SOURCES
    src/main.cpp
)

add_executable(mlvc_demo ${MLVC_DEMO_SOURCES})

target_include_directories(mlvc_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(mlvc_demo PRIVATE
    msrtc_rans
    nlohmann_json::nlohmann_json
    ${ONNXRUNTIME_LIBRARY}
)

# Keep the binary relocatable: look for libonnxruntime next to the executable
set_target_properties(mlvc_demo PROPERTIES
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN"
)

install(TARGETS mlvc_demo RUNTIME DESTINATION bin)
