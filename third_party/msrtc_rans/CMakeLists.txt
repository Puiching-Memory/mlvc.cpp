# Vendored copy of microsoft/mlvc packages/msrtc_rans (MIT License).
# Original project: https://github.com/microsoft/mlvc

cmake_minimum_required(VERSION 3.23)
project(msrtc_rans LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(msrtc_rans STATIC src/EntropyCoder.cpp)
target_include_directories(
    msrtc_rans
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/private/include
)
