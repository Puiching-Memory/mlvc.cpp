add_library(mlvc_codec_objects OBJECT src/pipeline.cpp)
set_target_properties(mlvc_codec_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_codec_objects PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mlvc_codec_objects PRIVATE
    mlvc_core
    mlvc_runtime
    msrtc_rans
    mlvc_model_proto)
mlvc_configure_target(mlvc_codec_objects)
mlvc_enable_ipo(mlvc_codec_objects)

add_library(mlvc_codec SHARED
    src/c_api.cpp
    $<TARGET_OBJECTS:mlvc_codec_objects>
    $<TARGET_OBJECTS:mlvc_backend_registry>)
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
        "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(mlvc_codec PRIVATE
    mlvc_core
    mlvc_runtime
    mlvc_backend
    msrtc_rans
    mlvc_model_proto)
mlvc_configure_target(mlvc_codec)
mlvc_enable_ipo(mlvc_codec)
if(UNIX AND NOT APPLE)
    target_link_options(mlvc_codec PRIVATE
        "LINKER:--version-script=${PROJECT_SOURCE_DIR}/cmake/codec.exports.map")
    set_property(TARGET mlvc_codec APPEND PROPERTY LINK_DEPENDS
        "${PROJECT_SOURCE_DIR}/cmake/codec.exports.map")
elseif(WIN32)
    set_target_properties(mlvc_codec PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

install(TARGETS mlvc_codec
    EXPORT mlvc_codecTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
