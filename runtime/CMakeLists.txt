add_library(mlvc_runtime INTERFACE)
target_include_directories(mlvc_runtime
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(mlvc_runtime INTERFACE mlvc_core)
