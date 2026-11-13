set(MLVC_CPU_SOURCES
    src/cpu_backend.cpp
    src/cpu_features.cpp
    src/execution_plan.cpp
    src/fusion_matcher.cpp
    src/kernels_scalar.cpp
    src/thread_pool.cpp
)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _mlvc_cpu_arch)
set(_mlvc_cpu_arm64 FALSE)
set(_mlvc_cpu_x86_64 FALSE)
if(_mlvc_cpu_arch MATCHES "^(aarch64|arm64)$" OR
   ANDROID_ABI STREQUAL "arm64-v8a" OR
   OHOS_ARCH STREQUAL "arm64-v8a")
    set(_mlvc_cpu_arm64 TRUE)
elseif(_mlvc_cpu_arch MATCHES "^(x86_64|amd64)$")
    set(_mlvc_cpu_x86_64 TRUE)
endif()

if(_mlvc_cpu_arm64 AND NOT MSVC)
    enable_language(ASM)
    set_source_files_properties(src/kernels_arm64_fp16.cpp PROPERTIES
        COMPILE_OPTIONS "-march=armv8.2-a+fp16"
    )
    set_source_files_properties(src/kernels_arm64_fp16fml_asm.S PROPERTIES
        COMPILE_OPTIONS "-march=armv8.2-a+fp16fml"
    )
    list(APPEND MLVC_CPU_SOURCES
        src/kernels_arm64_neon.cpp
        src/kernels_arm64_neon_asm.S
        src/kernels_arm64_fp16.cpp
        src/kernels_arm64_fp16fml.cpp
        src/kernels_arm64_fp16fml_asm.S
    )
elseif(_mlvc_cpu_x86_64)
    if(MSVC)
        set_source_files_properties(src/kernels_x86_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2"
        )
    else()
        set_source_files_properties(src/kernels_x86_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx2;-mfma;-mf16c"
        )
    endif()
    list(APPEND MLVC_CPU_SOURCES src/kernels_x86_avx2.cpp)
endif()

add_library(mlvc_backend_cpu STATIC ${MLVC_CPU_SOURCES})
set_target_properties(mlvc_backend_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_backend_cpu PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(mlvc_backend_cpu
    PUBLIC
        mlvc_runtime
    PRIVATE
        mlvc_core
        Threads::Threads
)
if(_mlvc_cpu_arm64 AND NOT MSVC)
    target_compile_definitions(mlvc_backend_cpu PRIVATE MLVC_CPU_ENABLE_ARM64=1)
elseif(_mlvc_cpu_x86_64)
    target_compile_definitions(mlvc_backend_cpu PRIVATE MLVC_CPU_ENABLE_X86_AVX2=1)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(mlvc_backend_cpu PRIVATE "$<$<CONFIG:Release>:-O3>")
endif()

mlvc_configure_target(mlvc_backend_cpu)
mlvc_enable_ipo(mlvc_backend_cpu)
