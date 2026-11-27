set(MLVC_WEBGPU_SOURCES
    execution_plan.cpp
    fusion_matcher.cpp
    shaders.cpp
    webgpu_backend.cpp
    wgpu_context.cpp
)

add_library(mlvc_backend_webgpu STATIC ${MLVC_WEBGPU_SOURCES})
set_target_properties(mlvc_backend_webgpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_backend_webgpu
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_link_libraries(mlvc_backend_webgpu
    PUBLIC
        mlvc_runtime
    PRIVATE
        mlvc_core
)

if(EMSCRIPTEN)
    target_compile_options(mlvc_backend_webgpu PUBLIC "--use-port=emdawnwebgpu")
    target_link_options(mlvc_backend_webgpu PUBLIC "--use-port=emdawnwebgpu")
else()
    if(EXISTS "${MLVC_DAWN_DIR}/webgpu.h")
        target_include_directories(mlvc_backend_webgpu SYSTEM PUBLIC "${MLVC_DAWN_DIR}")
        target_compile_definitions(mlvc_backend_webgpu PUBLIC MLVC_WEBGPU_FLAT_HEADER=1)
    elseif(EXISTS "${MLVC_DAWN_DIR}/include/webgpu/webgpu.h")
        target_include_directories(mlvc_backend_webgpu SYSTEM PUBLIC "${MLVC_DAWN_DIR}/include")
    else()
        message(FATAL_ERROR
            "Could not find webgpu.h under MLVC_DAWN_DIR='${MLVC_DAWN_DIR}'")
    endif()

    find_library(MLVC_DAWN_LIBRARY
        NAMES webgpu_dawn dawn
        HINTS "${MLVC_DAWN_DIR}" "${MLVC_DAWN_DIR}/lib"
        NO_DEFAULT_PATH
        REQUIRED)
    target_link_libraries(mlvc_backend_webgpu PUBLIC "${MLVC_DAWN_LIBRARY}")

    if(WIN32 AND EXISTS "${MLVC_DAWN_DIR}/webgpu_dawn.dll")
        set(_mlvc_dawn_dll_dst "${CMAKE_BINARY_DIR}/bin/webgpu_dawn.dll")
        add_custom_command(
            OUTPUT "${_mlvc_dawn_dll_dst}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/bin"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${MLVC_DAWN_DIR}/webgpu_dawn.dll" "${_mlvc_dawn_dll_dst}"
            DEPENDS "${MLVC_DAWN_DIR}/webgpu_dawn.dll"
            COMMENT "Copying webgpu_dawn.dll to ${CMAKE_BINARY_DIR}/bin")
        add_custom_target(mlvc_copy_dawn_dll ALL DEPENDS "${_mlvc_dawn_dll_dst}")
        add_dependencies(mlvc_backend_webgpu mlvc_copy_dawn_dll)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(mlvc_backend_webgpu PRIVATE "$<$<CONFIG:Release>:-O3>")
endif()

mlvc_configure_target(mlvc_backend_webgpu)
mlvc_enable_ipo(mlvc_backend_webgpu)
