add_executable(mlvc_tool_demo cli/main.cpp)
set_target_properties(mlvc_tool_demo PROPERTIES
    OUTPUT_NAME mlvc_demo
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN/../lib")
target_include_directories(mlvc_tool_demo PRIVATE
    "${PROJECT_SOURCE_DIR}/public/include"
    "${PROJECT_SOURCE_DIR}/codec/include"
    "${PROJECT_SOURCE_DIR}/core/include"
    "${PROJECT_SOURCE_DIR}/runtime/include")
target_link_libraries(mlvc_tool_demo PRIVATE mlvc_codec)
target_link_options(mlvc_tool_demo PRIVATE "LINKER:--disable-new-dtags")
mlvc_configure_target(mlvc_tool_demo)
mlvc_enable_ipo(mlvc_tool_demo)

add_executable(mlvc_tool_benchmark benchmark/main.cpp)
set_target_properties(mlvc_tool_benchmark PROPERTIES
    OUTPUT_NAME mlvc_backend_bench
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN/../lib")
target_include_directories(mlvc_tool_benchmark PRIVATE
    "${PROJECT_SOURCE_DIR}/core/include"
    "${PROJECT_SOURCE_DIR}/runtime/include"
    "${NLOHMANN_JSON_DIR}/single_include")
target_link_libraries(mlvc_tool_benchmark PRIVATE
    mlvc_core
    mlvc_runtime
    mlvc_backend_driver_cubin
    nlohmann_json::nlohmann_json)
target_link_options(mlvc_tool_benchmark PRIVATE "LINKER:--disable-new-dtags")
mlvc_configure_target(mlvc_tool_benchmark)
mlvc_enable_ipo(mlvc_tool_benchmark)

# Resolve the driver SONAME against the toolkit stub on driverless build hosts.
target_link_options(mlvc_tool_demo PRIVATE
    "LINKER:-rpath-link,${CUDAToolkit_LIBRARY_DIR}/stubs")
target_link_options(mlvc_tool_benchmark PRIVATE
    "LINKER:-rpath-link,${CUDAToolkit_LIBRARY_DIR}/stubs")

install(TARGETS mlvc_tool_demo mlvc_tool_benchmark
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
