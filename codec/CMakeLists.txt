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
    mlvc_backend_selected
    msrtc_rans
    nlohmann_json::nlohmann_json
    ${CMAKE_DL_LIBS})
mlvc_configure_target(mlvc_codec)
mlvc_enable_ipo(mlvc_codec)

install(TARGETS mlvc_codec
    EXPORT mlvc_codecTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
