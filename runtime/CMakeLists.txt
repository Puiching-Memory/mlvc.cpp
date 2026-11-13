add_library(mlvc_backend_registry OBJECT
    src/backend_registry.cpp)
set_target_properties(mlvc_backend_registry PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_backend_registry
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(mlvc_backend_registry
    PUBLIC
        mlvc_core)
mlvc_configure_target(mlvc_backend_registry)
mlvc_enable_ipo(mlvc_backend_registry)

add_library(mlvc_runtime STATIC
    src/aot_graph.cpp)
set_target_properties(mlvc_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(mlvc_runtime
    PUBLIC
        mlvc_core
        mlvc_backend_registry
    PRIVATE
        mlvc_model_proto)
mlvc_configure_target(mlvc_runtime)
mlvc_enable_ipo(mlvc_runtime)
