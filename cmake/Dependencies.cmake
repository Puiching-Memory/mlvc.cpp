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

set(MLVC_CUTLASS_DIR "${PROJECT_SOURCE_DIR}/third_party/cutlass")
if(NOT EXISTS "${MLVC_CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR
        "CUTLASS source submodule is missing. Run: "
        "git submodule update --init --recursive --depth 1")
endif()
find_package(CUDAToolkit 13.3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
enable_language(ASM)
