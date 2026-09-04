add_library(mlvc_backend_tensorrt STATIC src/backend.cpp)
set_target_properties(mlvc_backend_tensorrt PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_backend_tensorrt PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
    "${CUDAToolkit_INCLUDE_DIRS}"
    "${NLOHMANN_JSON_DIR}/single_include")
target_link_libraries(mlvc_backend_tensorrt PRIVATE
    mlvc_runtime
    "${NVINFER_LIBRARY}"
    "${NVONNXPARSER_LIBRARY}"
    CUDA::cudart)
mlvc_enable_ipo(mlvc_backend_tensorrt)

file(GLOB _tensorrt_runtime_libraries
    "${TENSORRT_LIBRARY_DIR}/libnvinfer.so*"
    "${TENSORRT_LIBRARY_DIR}/libnvinfer_builder_resource_*.so*"
    "${TENSORRT_LIBRARY_DIR}/libnvonnxparser.so*")
install(FILES ${_tensorrt_runtime_libraries}
    DESTINATION "${CMAKE_INSTALL_LIBDIR}")
