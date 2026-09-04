add_library(mlvc_backend_onnxruntime STATIC src/backend.cpp)
set_target_properties(mlvc_backend_onnxruntime PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_backend_onnxruntime PRIVATE
    "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(mlvc_backend_onnxruntime PRIVATE
    mlvc_runtime
    "${ONNXRUNTIME_LIBRARY}")
mlvc_enable_ipo(mlvc_backend_onnxruntime)

file(GLOB _onnxruntime_runtime_libraries
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*"
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_shared.so*"
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_cuda.so*"
    "${ONNXRUNTIME_ROOT}/lib/libcudnn*.so*")
install(FILES ${_onnxruntime_runtime_libraries}
    DESTINATION "${CMAKE_INSTALL_LIBDIR}")
