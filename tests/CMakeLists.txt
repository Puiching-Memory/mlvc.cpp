add_executable(mlvc_codec_tests unit/codec_tests.cpp)
target_include_directories(mlvc_codec_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/core/include")
target_link_libraries(mlvc_codec_tests PRIVATE mlvc_core)
target_link_options(mlvc_codec_tests PRIVATE "LINKER:--disable-new-dtags")
mlvc_enable_ipo(mlvc_codec_tests)

add_executable(mlvc_c_api_tests unit/c_api_tests.cpp)
target_include_directories(mlvc_c_api_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/public/include")
target_link_libraries(mlvc_c_api_tests PRIVATE mlvc_codec)
target_link_options(mlvc_c_api_tests PRIVATE "LINKER:--disable-new-dtags")
mlvc_enable_ipo(mlvc_c_api_tests)

add_test(NAME mlvc_c_api COMMAND mlvc_c_api_tests)

add_executable(mlvc_public_header_tests unit/public_header_tests.c)
target_include_directories(mlvc_public_header_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/public/include")
add_test(NAME mlvc_public_header COMMAND mlvc_public_header_tests)

if(EXISTS "${MLVC_TEST_ASSETS_DIR}/benchmarks/gray-q21/encoder/case.json"
   AND EXISTS "${MLVC_MODEL_ROOT}/mlvc-psnr-v1/640x368/metadata.json")
    add_test(NAME mlvc_codec_reference
        COMMAND mlvc_codec_tests "${MLVC_TEST_ASSETS_DIR}")
else()
    message(STATUS
        "Portable codec fixtures not found under ${MLVC_TEST_ASSETS_DIR}; "
        "mlvc_codec_reference will not be registered")
endif()

set(_mlvc_conformance_script
    "${PROJECT_SOURCE_DIR}/tests/conformance/run_codec_conformance.py")
if(EXISTS "${MLVC_TEST_ASSETS_DIR}/references/mlvc-psnr-v1/gray-q21-2f/reference.json"
   AND EXISTS "${MLVC_TEST_ASSETS_DIR}/references/mlvc-s-psnr-v1/gray-q21-2f/reference.json")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    foreach(_mlvc_profile mlvc-psnr-v1 mlvc-s-psnr-v1)
        set(_mlvc_model_dir
            "${MLVC_MODEL_ROOT}/${_mlvc_profile}/640x368")
        set(_mlvc_reference_dir
            "${MLVC_TEST_ASSETS_DIR}/references/${_mlvc_profile}/gray-q21-2f")
        if(MLVC_SELECTED_BACKEND STREQUAL "driver_cubin")
            set(_mlvc_conformance_args
                --binary "$<TARGET_FILE:mlvc_tool_demo>"
                --model-dir "embedded:${_mlvc_profile}"
                --bundle-dir "${_mlvc_model_dir}"
                --reference-dir "${_mlvc_reference_dir}")
        else()
            set(_mlvc_conformance_args
                --binary "$<TARGET_FILE:mlvc_tool_demo>"
                --model-dir "${_mlvc_model_dir}"
                --reference-dir "${_mlvc_reference_dir}")
        endif()
        if(MLVC_SELECTED_BACKEND STREQUAL "tensorrt"
           AND MLVC_TEST_ENGINE_CACHE_DIR)
            list(APPEND _mlvc_conformance_args
                --engine-cache-dir "${MLVC_TEST_ENGINE_CACHE_DIR}")
        endif()
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
