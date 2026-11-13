cmake_minimum_required(VERSION 3.23)
project(mlvc_cpp VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

set(MLVC_BACKEND "AUTO" CACHE STRING
    "Inference backend to build (AUTO, cpu, driver_cubin, all, none)")
set_property(CACHE MLVC_BACKEND PROPERTY STRINGS AUTO cpu driver_cubin all none)

set(_mlvc_is_linux_x86_64 FALSE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
   CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(_mlvc_is_linux_x86_64 TRUE)
endif()

if(MLVC_BACKEND STREQUAL "AUTO")
    if(_mlvc_is_linux_x86_64)
        set(_mlvc_resolved_backend "driver_cubin")
    else()
        set(_mlvc_resolved_backend "cpu")
    endif()
    set(MLVC_BACKEND "${_mlvc_resolved_backend}" CACHE STRING
        "Inference backend to build (AUTO, cpu, driver_cubin, all, none)" FORCE)
endif()

if(NOT MLVC_BACKEND MATCHES "^(cpu|driver_cubin|all|none)$")
    message(FATAL_ERROR
        "Unsupported MLVC_BACKEND='${MLVC_BACKEND}'. "
        "Supported values: AUTO, cpu, driver_cubin, all, none.")
endif()
if(MLVC_BACKEND MATCHES "^(driver_cubin|all)$" AND NOT _mlvc_is_linux_x86_64)
    message(FATAL_ERROR
        "MLVC_BACKEND=${MLVC_BACKEND} requires a Linux x86_64 NVIDIA host "
        "(current target: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}). "
        "Configure with -DMLVC_BACKEND=cpu or -DMLVC_BACKEND=none.")
endif()
message(STATUS "MLVC inference backend: ${MLVC_BACKEND}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(_mlvc_default_embed_models OFF)
if(MLVC_BACKEND MATCHES "^(driver_cubin|all)$")
    set(_mlvc_default_embed_models ON)
endif()
option(MLVC_ENABLE_IPO "Enable interprocedural optimization in Release builds" ON)
option(MLVC_EMBED_MODELS
    "Embed canonical model bundles into the core library"
    ${_mlvc_default_embed_models})
option(MLVC_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)
option(MLVC_ENABLE_SANITIZERS
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
set(MLVC_TEST_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models/fixtures" CACHE PATH
    "Local MLVC model/reference fixtures used by tests")
set(MLVC_MODEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/models/canonical" CACHE PATH
    "Canonical model bundle root")
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
        get_target_property(_mlvc_target_type "${target}" TYPE)
        # CMake's WINDOWS_EXPORT_ALL_SYMBOLS scanner cannot read MSVC /GL LTCG objects.
        if(MSVC AND (_mlvc_target_type STREQUAL "SHARED_LIBRARY"
                     OR _mlvc_target_type STREQUAL "OBJECT_LIBRARY"))
            return()
        endif()
        set_property(TARGET "${target}"
            PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endfunction()

add_library(mlvc_project_options INTERFACE)
target_compile_options(mlvc_project_options INTERFACE
    "$<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Wformat=2;-Wshadow;-Wundef>"
    "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Wformat=2;-Wshadow;-Wundef>"
    "$<$<COMPILE_LANG_AND_ID:C,MSVC>:/W4;/utf-8>"
    "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4;/utf-8>")
target_compile_definitions(mlvc_project_options INTERFACE
    "$<$<C_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>"
    "$<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>")
if(MLVC_WARNINGS_AS_ERRORS)
    target_compile_options(mlvc_project_options INTERFACE
        "$<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-Werror>"
        "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Werror>"
        "$<$<COMPILE_LANG_AND_ID:C,MSVC>:/WX>"
        "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/WX>")
endif()
if(MLVC_ENABLE_SANITIZERS)
    if(NOT CMAKE_C_COMPILER_ID MATCHES "GNU|Clang" OR
       NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "MLVC sanitizers require GCC or Clang")
    endif()
    target_compile_options(mlvc_project_options INTERFACE
        -fno-omit-frame-pointer -fno-sanitize-recover=all
        -fsanitize=address,undefined)
    target_link_options(mlvc_project_options INTERFACE
        -fno-sanitize-recover=all -fsanitize=address,undefined)
endif()

function(mlvc_configure_target target)
    target_link_libraries("${target}" PRIVATE mlvc_project_options)
endfunction()

if(OHOS AND (CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    # Huawei ohos.toolchain.cmake sets external toolchain paths that emit
    # unused --gcc-toolchain warnings on C/C++ compile steps under Clang 15.
    add_compile_options(-Wno-unused-command-line-argument)
endif()

include(cmake/Dependencies.cmake)

if(WIN32)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

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
