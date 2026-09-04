cmake_minimum_required(VERSION 3.23)
project(mlvc_installed_consumer LANGUAGES C)

find_package(mlvc_codec CONFIG REQUIRED)

add_executable(mlvc_installed_consumer main.c)
target_link_libraries(mlvc_installed_consumer PRIVATE mlvc::mlvc_codec)
