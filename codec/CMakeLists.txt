add_library(mlvc_codec SHARED
    src/c_api.cpp
    src/pipeline.cpp)
set_target_properties(mlvc_codec PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME mlvc_codec
    POSITION_INDEPENDENT_CODE ON
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN")
target_include_directories(mlvc_codec
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/public/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/core/src")
target_link_libraries(mlvc_codec PRIVATE
    mlvc_core
    mlvc_runtime
    mlvc_backend_driver_cubin
    msrtc_rans
    mlvc_model_proto)
mlvc_configure_target(mlvc_codec)
mlvc_enable_ipo(mlvc_codec)
target_link_options(mlvc_codec PRIVATE
    "LINKER:--version-script=${PROJECT_SOURCE_DIR}/cmake/codec.exports.map")
set_property(TARGET mlvc_codec APPEND PROPERTY LINK_DEPENDS
    "${PROJECT_SOURCE_DIR}/cmake/codec.exports.map")

install(TARGETS mlvc_codec
    EXPORT mlvc_codecTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
