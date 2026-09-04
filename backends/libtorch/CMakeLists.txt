add_library(mlvc_backend_libtorch STATIC src/backend.cpp)
set_target_properties(mlvc_backend_libtorch PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_backend_libtorch PRIVATE
    ${TORCH_INCLUDE_DIRS})
target_link_libraries(mlvc_backend_libtorch PRIVATE
    mlvc_runtime
    ${TORCH_LIBRARIES})
mlvc_configure_target(mlvc_backend_libtorch)
mlvc_enable_ipo(mlvc_backend_libtorch)

file(GLOB _libtorch_runtime_libraries "${LIBTORCH_ROOT}/lib/*.so*")
install(FILES ${_libtorch_runtime_libraries}
    DESTINATION "${CMAKE_INSTALL_LIBDIR}")
