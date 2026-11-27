# Project and third-party dependencies are resolved in one place.  Backend
# subdirectories consume the resulting imported targets and *_ROOT variables;
# they do not perform global dependency discovery themselves.

set(MLVC_UPSTREAM_DIR "${PROJECT_SOURCE_DIR}/third_party/mlvc")
set(MLVC_PROFILE_FILE "${PROJECT_SOURCE_DIR}/tools/python/mlvc_tools/profiles.json")

if(NOT EXISTS "${MLVC_UPSTREAM_DIR}/packages/msrtc_rans/CMakeLists.txt")
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
if(TARGET msrtc_rans)
    get_target_property(_mlvc_rans_inc msrtc_rans INTERFACE_INCLUDE_DIRECTORIES)
    if(_mlvc_rans_inc)
        set_target_properties(msrtc_rans PROPERTIES
            INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_mlvc_rans_inc}")
    endif()
endif()

include("${PROJECT_SOURCE_DIR}/cmake/ModelProto.cmake")

if(MLVC_BACKEND MATCHES "^(driver_cubin|all)$")
    set(MLVC_CUTLASS_DIR "${PROJECT_SOURCE_DIR}/third_party/cutlass")
    if(NOT EXISTS "${MLVC_CUTLASS_DIR}/include/cutlass/cutlass.h")
        message(FATAL_ERROR
            "CUTLASS source submodule is missing. Run: "
            "git submodule update --init --recursive --depth 1")
    endif()
    find_package(CUDAToolkit 12.0 REQUIRED)
endif()

if(MLVC_BACKEND MATCHES "^(cpu|webgpu_cpu|all)$")
    find_package(Threads REQUIRED)
endif()

if(MLVC_BACKEND MATCHES "^(webgpu|webgpu_cpu)$" AND NOT EMSCRIPTEN)
    set(_mlvc_default_dawn_dir "")
    if(EXISTS "${PROJECT_SOURCE_DIR}/build/tools/dawn/dawn-x64/webgpu.h")
        set(_mlvc_default_dawn_dir "${PROJECT_SOURCE_DIR}/build/tools/dawn/dawn-x64")
    endif()
    set(MLVC_DAWN_DIR "${_mlvc_default_dawn_dir}" CACHE PATH
        "Directory containing Dawn webgpu.h and webgpu_dawn library")
    if(NOT MLVC_DAWN_DIR OR NOT EXISTS "${MLVC_DAWN_DIR}")
        message(FATAL_ERROR
            "MLVC_BACKEND=${MLVC_BACKEND} on native targets requires Dawn. "
            "Pass -DMLVC_DAWN_DIR=/path/to/dawn containing webgpu.h and webgpu_dawn.")
    endif()
endif()

set(_mlvc_python_protobuf_ok FALSE)
if(MLVC_EMBED_MODELS)
    enable_language(ASM)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
elseif(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
endif()
if(Python3_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
            "import google.protobuf; assert google.protobuf.__version__ == '6.33.5'"
        RESULT_VARIABLE _mlvc_python_proto_result
        ERROR_QUIET)
    if(_mlvc_python_proto_result EQUAL 0)
        set(_mlvc_python_protobuf_ok TRUE)
    elseif(MLVC_EMBED_MODELS)
        message(FATAL_ERROR
            "Model embedding requires Python protobuf==6.33.5. Activate the model environment "
            "or set Python3_EXECUTABLE to its interpreter.")
    endif()
endif()
