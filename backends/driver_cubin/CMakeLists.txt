set(MLVC_DRIVER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(MLVC_DRIVER_KERNEL_DIR "${MLVC_DRIVER_SOURCE_DIR}/kernels")
set(MLVC_DRIVER_TOOL_DIR "${MLVC_DRIVER_SOURCE_DIR}/tools")
set(MLVC_DRIVER_GENERATED_DIR
    "${PROJECT_BINARY_DIR}/generated/driver_cubin")
set(MLVC_DRIVER_FATBIN_DIR
    "${MLVC_DRIVER_GENERATED_DIR}/fatbin")
set(MLVC_DRIVER_FATBIN
    "${MLVC_DRIVER_FATBIN_DIR}/mlvc_driver_kernels.fatbin")
set(MLVC_DRIVER_FATBIN_EMBED_DIR
    "${MLVC_DRIVER_GENERATED_DIR}/embed")
set(MLVC_DRIVER_FATBIN_SOURCE
    "${MLVC_DRIVER_FATBIN_EMBED_DIR}/mlvc_driver_kernels_fatbin.cpp")

set(MLVC_DRIVER_KERNEL_SOURCES
    "${MLVC_DRIVER_KERNEL_DIR}/module.cu"
    "${MLVC_DRIVER_KERNEL_DIR}/activation.cu"
    "${MLVC_DRIVER_KERNEL_DIR}/common.cuh"
    "${MLVC_DRIVER_KERNEL_DIR}/cutlass.cu"
    "${MLVC_DRIVER_KERNEL_DIR}/depthwise_conv.cu"
    "${MLVC_DRIVER_KERNEL_DIR}/elementwise.cu"
    "${MLVC_DRIVER_KERNEL_DIR}/generic_conv.cu"
    "${MLVC_DRIVER_KERNEL_DIR}/layout.cu"
    "${MLVC_DRIVER_KERNEL_DIR}/mma_common.cuh"
    "${MLVC_DRIVER_KERNEL_DIR}/pointwise_conv.cu"
    "${MLVC_DRIVER_KERNEL_DIR}/probe.cu"
    "${MLVC_DRIVER_KERNEL_DIR}/quantization_tail.cu"
    "${MLVC_DRIVER_KERNEL_DIR}/recurrent_feature.cu"
    "${MLVC_DRIVER_KERNEL_DIR}/spatial_conv.cu"
    "${MLVC_DRIVER_KERNEL_DIR}/tensor_ops.cu"
    "${MLVC_DRIVER_SOURCE_DIR}/include/mlvc/driver_cubin/cutlass.hpp")
file(MAKE_DIRECTORY "${MLVC_DRIVER_FATBIN_DIR}")
add_custom_command(
    OUTPUT "${MLVC_DRIVER_FATBIN}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MLVC_DRIVER_FATBIN_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E env
        "NVCC=${CUDAToolkit_NVCC_EXECUTABLE}"
        "${MLVC_DRIVER_TOOL_DIR}/build_fatbin.sh"
        --output "${MLVC_DRIVER_FATBIN}"
    DEPENDS
        ${MLVC_DRIVER_KERNEL_SOURCES}
        "${MLVC_DRIVER_TOOL_DIR}/build_fatbin.sh"
    COMMENT "Building driver-cubin CUDA fatbin"
    VERBATIM)
add_custom_target(mlvc_driver_fatbin DEPENDS "${MLVC_DRIVER_FATBIN}")

file(MAKE_DIRECTORY "${MLVC_DRIVER_FATBIN_EMBED_DIR}")
add_custom_command(
    OUTPUT "${MLVC_DRIVER_FATBIN_SOURCE}"
    COMMAND "${CMAKE_COMMAND}"
        "-DINPUT=${MLVC_DRIVER_FATBIN}"
        "-DOUTPUT=${MLVC_DRIVER_FATBIN_SOURCE}"
        -DSYMBOL=mlvc_driver_kernels_fatbin
        -P "${PROJECT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
    DEPENDS
        mlvc_driver_fatbin
        "${PROJECT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
    COMMENT "Embedding driver-cubin fatbin"
    VERBATIM)

add_library(mlvc_driver_kernels STATIC "${MLVC_DRIVER_FATBIN_SOURCE}")
set_target_properties(mlvc_driver_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
add_dependencies(mlvc_driver_kernels mlvc_driver_fatbin)
mlvc_enable_ipo(mlvc_driver_kernels)

add_library(mlvc_driver STATIC src/driver.cpp)
set_target_properties(mlvc_driver PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_driver PUBLIC
    "${MLVC_DRIVER_SOURCE_DIR}/include"
    "${CUDAToolkit_INCLUDE_DIRS}")
target_link_libraries(mlvc_driver PUBLIC CUDA::cuda_driver)
mlvc_enable_ipo(mlvc_driver)

add_executable(mlvc_driver_probe tools/probe_main.cpp)
target_link_libraries(mlvc_driver_probe PRIVATE
    mlvc_driver mlvc_driver_kernels)
target_link_options(mlvc_driver_probe PRIVATE "LINKER:--disable-new-dtags")
mlvc_enable_ipo(mlvc_driver_probe)

add_library(mlvc_backend_driver_cubin STATIC
    src/backend.cpp
    src/cuda_graph_executor.cpp
    src/execution_plan.cpp
    src/fusion_registry.cpp
    src/graph_ir.cpp
    src/graph_validator.cpp
    src/kernel_registry.cpp
    src/memory_planner.cpp
    src/model_loader.cpp)
set_target_properties(mlvc_backend_driver_cubin PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_backend_driver_cubin PRIVATE
    "${MLVC_DRIVER_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/core/src"
    "${NLOHMANN_JSON_DIR}/single_include")
target_link_libraries(mlvc_backend_driver_cubin PRIVATE
    mlvc_runtime
    mlvc_driver
    mlvc_driver_kernels
    nlohmann_json::nlohmann_json)
mlvc_enable_ipo(mlvc_backend_driver_cubin)

if(BUILD_TESTING)
    add_test(NAME mlvc_driver_cubin_probe
        COMMAND mlvc_driver_probe --iterations 100)
endif()

install(TARGETS mlvc_driver_probe RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
