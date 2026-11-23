add_executable(mlvc_tool_demo cli/main.cpp)
set_target_properties(mlvc_tool_demo PROPERTIES
    OUTPUT_NAME mlvc_demo
    BUILD_RPATH "\$ORIGIN"
    INSTALL_RPATH "\$ORIGIN/../lib")
target_include_directories(mlvc_tool_demo PRIVATE
    "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(mlvc_tool_demo PRIVATE mlvc_codec)
if(UNIX AND NOT APPLE)
    target_link_options(mlvc_tool_demo PRIVATE "LINKER:--disable-new-dtags")
endif()
mlvc_configure_target(mlvc_tool_demo)
mlvc_enable_ipo(mlvc_tool_demo)

install(TARGETS mlvc_tool_demo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
