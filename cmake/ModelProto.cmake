# Build-host protoc and deployed Lite runtime come from the same pinned source.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(_mlvc_previous_visibility "${CMAKE_CXX_VISIBILITY_PRESET}")
set(_mlvc_previous_inline_visibility "${CMAKE_VISIBILITY_INLINES_HIDDEN}")
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/abseil"
                 "${PROJECT_BINARY_DIR}/_deps/abseil" EXCLUDE_FROM_ALL)
set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(protobuf_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
# protoc's Rust generator requires upb even when deployed messages use Lite.
# This is a build-host dependency only; the codec links libprotobuf-lite.
set(protobuf_BUILD_LIBUPB ON CACHE BOOL "" FORCE)
set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
set(protobuf_INSTALL OFF CACHE BOOL "" FORCE)
set(protobuf_LOCAL_DEPENDENCIES_ONLY ON CACHE BOOL "" FORCE)
add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/protobuf"
                 "${PROJECT_BINARY_DIR}/_deps/protobuf" EXCLUDE_FROM_ALL)
# RepeatedField container annotations are emitted only by sanitizer-instrumented
# code. Instrument the runtime together with the generated messages, otherwise
# mixing instrumented and plain objects makes ASan report bogus
# container-overflow while parsing a message. AddressSanitizer only: UBSan turns
# Abseil's constexpr function-pointer comparison into a hard compile error.
if(MLVC_ENABLE_SANITIZERS)
    if(TARGET libprotobuf-lite)
        target_compile_options(libprotobuf-lite PRIVATE
            -fno-omit-frame-pointer -fsanitize=address)
        target_link_options(libprotobuf-lite PRIVATE -fsanitize=address)
    endif()
endif()
if(_mlvc_previous_visibility STREQUAL "")
    unset(CMAKE_CXX_VISIBILITY_PRESET)
else()
    set(CMAKE_CXX_VISIBILITY_PRESET "${_mlvc_previous_visibility}")
endif()
if(_mlvc_previous_inline_visibility STREQUAL "")
    unset(CMAKE_VISIBILITY_INLINES_HIDDEN)
else()
    set(CMAKE_VISIBILITY_INLINES_HIDDEN "${_mlvc_previous_inline_visibility}")
endif()

set(MLVC_PROTO_DIR "${PROJECT_BINARY_DIR}/generated/proto")
file(MAKE_DIRECTORY "${MLVC_PROTO_DIR}")
set(MLVC_PROTO_SCHEMA "${PROJECT_SOURCE_DIR}/proto/mlvc_model.proto")
add_custom_command(
    OUTPUT "${MLVC_PROTO_DIR}/mlvc_model.pb.cc" "${MLVC_PROTO_DIR}/mlvc_model.pb.h"
    COMMAND $<TARGET_FILE:protobuf::protoc>
        "--proto_path=${PROJECT_SOURCE_DIR}/proto" "--cpp_out=${MLVC_PROTO_DIR}"
        "${MLVC_PROTO_SCHEMA}"
    DEPENDS protobuf::protoc "${MLVC_PROTO_SCHEMA}"
    VERBATIM)
add_library(mlvc_model_proto STATIC "${MLVC_PROTO_DIR}/mlvc_model.pb.cc")
target_include_directories(mlvc_model_proto SYSTEM PUBLIC "${MLVC_PROTO_DIR}")
target_link_libraries(mlvc_model_proto PUBLIC protobuf::libprotobuf-lite)
# The Lite runtime includes utf8_validity.h from its private utf8_range
# dependency, and that include directory is only directory-scoped inside the
# Protobuf build tree. Propagate it to every consumer of generated messages.
if(TARGET utf8_validity)
    target_link_libraries(mlvc_model_proto PUBLIC utf8_validity)
else()
    target_include_directories(mlvc_model_proto SYSTEM PUBLIC
        "${PROJECT_SOURCE_DIR}/third_party/protobuf/third_party/utf8_range")
endif()
set_target_properties(mlvc_model_proto PROPERTIES
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
if(MLVC_ENABLE_SANITIZERS)
    target_compile_options(mlvc_model_proto PRIVATE
        -fno-omit-frame-pointer -fsanitize=address)
    target_link_options(mlvc_model_proto PRIVATE -fsanitize=address)
endif()

# Python bindings are checked in for standalone model tooling; regenerate from
# this target when the schema changes, using the exact same compiler.
add_custom_target(mlvc_generate_python_proto
    COMMAND $<TARGET_FILE:protobuf::protoc>
        "--proto_path=${PROJECT_SOURCE_DIR}/proto"
        "--python_out=${PROJECT_SOURCE_DIR}/tools/mlvc_proto" "${MLVC_PROTO_SCHEMA}"
    DEPENDS protobuf::protoc "${MLVC_PROTO_SCHEMA}"
    VERBATIM)
install(FILES "${PROJECT_SOURCE_DIR}/third_party/protobuf/LICENSE"
    DESTINATION "share/licenses/protobuf")
install(FILES "${PROJECT_SOURCE_DIR}/third_party/abseil/LICENSE"
    DESTINATION "share/licenses/abseil")
install(FILES "${PROJECT_SOURCE_DIR}/third_party/protobuf/third_party/utf8_range/LICENSE"
    DESTINATION "share/licenses/utf8_range")
