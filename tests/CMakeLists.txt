add_executable(mlvc_core_safety_tests unit/core_safety_tests.cpp)
target_include_directories(mlvc_core_safety_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/core/include")
target_link_libraries(mlvc_core_safety_tests PRIVATE mlvc_core)
mlvc_configure_target(mlvc_core_safety_tests)
mlvc_enable_ipo(mlvc_core_safety_tests)
add_test(NAME mlvc_core_safety COMMAND mlvc_core_safety_tests)

add_executable(mlvc_codec_tests unit/codec_tests.cpp)
target_include_directories(mlvc_codec_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/core/include")
target_link_libraries(mlvc_codec_tests PRIVATE mlvc_core)
target_link_options(mlvc_codec_tests PRIVATE "LINKER:--disable-new-dtags")
mlvc_configure_target(mlvc_codec_tests)
mlvc_enable_ipo(mlvc_codec_tests)

add_executable(mlvc_c_api_tests unit/c_api_tests.cpp)
target_include_directories(mlvc_c_api_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/public/include")
target_link_libraries(mlvc_c_api_tests PRIVATE mlvc_codec)
target_link_options(mlvc_c_api_tests PRIVATE "LINKER:--disable-new-dtags")
# Resolve the codec's transitive driver dependency on driverless build hosts.
target_link_options(mlvc_c_api_tests PRIVATE
    "LINKER:-rpath-link,${CUDAToolkit_LIBRARY_DIR}/stubs")
mlvc_configure_target(mlvc_c_api_tests)
mlvc_enable_ipo(mlvc_c_api_tests)

add_test(NAME mlvc_c_api COMMAND mlvc_c_api_tests)

add_executable(mlvc_public_header_tests unit/public_header_tests.c)
target_include_directories(mlvc_public_header_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/public/include")
mlvc_configure_target(mlvc_public_header_tests)
add_test(NAME mlvc_public_header COMMAND mlvc_public_header_tests)

add_executable(mlvc_model_proto_tests unit/model_proto_tests.cpp
    "${PROJECT_SOURCE_DIR}/backends/driver_cubin/src/graph_descriptor.cpp")
target_include_directories(mlvc_model_proto_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/core/src"
    "${PROJECT_SOURCE_DIR}/backends/driver_cubin/src")
target_link_libraries(mlvc_model_proto_tests PRIVATE mlvc_core mlvc_model_proto)
mlvc_configure_target(mlvc_model_proto_tests)
add_test(NAME mlvc_model_proto COMMAND mlvc_model_proto_tests)

add_test(NAME mlvc_model_package
    COMMAND "${Python3_EXECUTABLE}"
        "${PROJECT_SOURCE_DIR}/tests/unit/model_package_tests.py")

set(_mlvc_conformance_script
    "${PROJECT_SOURCE_DIR}/tests/conformance/run_codec_conformance.py")
if(EXISTS "${MLVC_TEST_ASSETS_DIR}/references/mlvc-psnr-v1/gray-q21-2f/reference.json"
   AND EXISTS "${MLVC_TEST_ASSETS_DIR}/references/mlvc-s-psnr-v1/gray-q21-2f/reference.json")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME mlvc_codec_reference
        COMMAND mlvc_codec_tests "${MLVC_TEST_ASSETS_DIR}" "${MLVC_MODEL_ROOT}")
    foreach(_mlvc_profile mlvc-psnr-v1 mlvc-s-psnr-v1)
        set(_mlvc_model_dir
            "${MLVC_MODEL_ROOT}/${_mlvc_profile}/640x368")
        set(_mlvc_reference_dir
            "${MLVC_TEST_ASSETS_DIR}/references/${_mlvc_profile}/gray-q21-2f")
        set(_mlvc_conformance_args
            --binary "$<TARGET_FILE:mlvc_tool_demo>"
            --model-dir "embedded:${_mlvc_profile}"
            --bundle-dir "${_mlvc_model_dir}"
            --reference-dir "${_mlvc_reference_dir}")
        add_test(NAME "mlvc_codec_conformance_${_mlvc_profile}"
            COMMAND "${Python3_EXECUTABLE}" "${_mlvc_conformance_script}"
                    ${_mlvc_conformance_args})
        set_tests_properties("mlvc_codec_conformance_${_mlvc_profile}"
            PROPERTIES LABELS "gpu;conformance" TIMEOUT 300)
    endforeach()
    add_custom_target(mlvc_codec_compatibility
        COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L conformance
        DEPENDS mlvc_tool_demo
        USES_TERMINAL
        COMMENT "Checking both MLVC profiles against official Python codec references")
else()
    message(STATUS
        "Codec conformance references not found under ${MLVC_TEST_ASSETS_DIR}; "
        "only portable codec tests will be registered")
    add_custom_target(mlvc_codec_compatibility
        COMMAND "${CMAKE_COMMAND}" -E echo
            "Missing MLVC and MLVC-S references under ${MLVC_TEST_ASSETS_DIR}"
        COMMAND "${CMAKE_COMMAND}" -E false
        USES_TERMINAL
        COMMENT "Codec compatibility references are required")
endif()
