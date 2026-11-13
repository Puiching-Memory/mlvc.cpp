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
    set(MLVC_KLEIDIAI_DIR "${PROJECT_SOURCE_DIR}/third_party/kleidiai")
    if(NOT EXISTS "${MLVC_KLEIDIAI_DIR}/kai/kai_common.h")
        message(FATAL_ERROR
            "Vendored KleidiAI headers are missing under ${MLVC_KLEIDIAI_DIR}")
    endif()
    add_library(mlvc_kleidiai_f16 OBJECT
        "${MLVC_KLEIDIAI_DIR}/kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.c"
        "${MLVC_KLEIDIAI_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.c"
    )
    set_target_properties(mlvc_kleidiai_f16 PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF
    )
    target_include_directories(mlvc_kleidiai_f16 PRIVATE "${MLVC_KLEIDIAI_DIR}")
    target_compile_options(mlvc_kleidiai_f16 PRIVATE -O3 -march=armv8.2-a+fp16 -w)

    set_source_files_properties(src/kernels_arm64_fp16.cpp PROPERTIES
        COMPILE_OPTIONS "-march=armv8.2-a+fp16"
        INCLUDE_DIRECTORIES "${MLVC_KLEIDIAI_DIR}"
    )
    set_source_files_properties(src/kernels_arm64_fp16fml.cpp PROPERTIES
        COMPILE_OPTIONS "-march=armv8.2-a+fp16fml"
    )
    list(APPEND MLVC_CPU_SOURCES
        src/kernels_arm64_fp16.cpp
        src/kernels_arm64_fp16fml.cpp
        $<TARGET_OBJECTS:mlvc_kleidiai_f16>
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
