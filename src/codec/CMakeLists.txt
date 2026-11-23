add_library(mlvc_codec_objects OBJECT pipeline.cpp)
set_target_properties(mlvc_codec_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_codec_objects PRIVATE
    "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(mlvc_codec_objects PRIVATE
    mlvc_core
    mlvc_runtime
    msrtc_rans
    mlvc_model_proto)
mlvc_configure_target(mlvc_codec_objects)
mlvc_enable_ipo(mlvc_codec_objects)

if(MLVC_BUILD_SHARED)
    set(_mlvc_codec_lib_type SHARED)
else()
    set(_mlvc_codec_lib_type STATIC)
endif()

add_library(mlvc_codec ${_mlvc_codec_lib_type}
    c_api.cpp
    $<TARGET_OBJECTS:mlvc_codec_objects>
    $<TARGET_OBJECTS:mlvc_backend_registry>)
set_target_properties(mlvc_codec PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME mlvc_codec
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN")
target_include_directories(mlvc_codec
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${PROJECT_SOURCE_DIR}/src")
if(MLVC_BUILD_SHARED)
    target_compile_definitions(mlvc_codec PRIVATE MLVC_CODEC_BUILDING=1)
else()
    target_compile_definitions(mlvc_codec PUBLIC MLVC_CODEC_STATIC=1)
endif()
target_link_libraries(mlvc_codec PRIVATE
    mlvc_core
    mlvc_runtime
    mlvc_backend
    msrtc_rans
    mlvc_model_proto)
mlvc_configure_target(mlvc_codec)
mlvc_enable_ipo(mlvc_codec)
if(MLVC_BUILD_SHARED AND UNIX AND NOT APPLE)
    target_link_options(mlvc_codec PRIVATE
        "LINKER:--version-script=${PROJECT_SOURCE_DIR}/cmake/codec.exports.map")
    set_property(TARGET mlvc_codec APPEND PROPERTY LINK_DEPENDS
        "${PROJECT_SOURCE_DIR}/cmake/codec.exports.map")
endif()

if(MLVC_BUILD_SHARED)
    install(TARGETS mlvc_codec
        EXPORT mlvc_codecTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
