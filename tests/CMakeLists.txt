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
if(UNIX AND NOT APPLE)
    target_link_options(mlvc_codec_tests PRIVATE "LINKER:--disable-new-dtags")
endif()
mlvc_configure_target(mlvc_codec_tests)
mlvc_enable_ipo(mlvc_codec_tests)

add_executable(mlvc_c_api_tests unit/c_api_tests.cpp)
target_include_directories(mlvc_c_api_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/public/include")
target_link_libraries(mlvc_c_api_tests PRIVATE mlvc_codec)
if(UNIX AND NOT APPLE)
    target_link_options(mlvc_c_api_tests PRIVATE "LINKER:--disable-new-dtags")
endif()
# Resolve the codec's transitive driver dependency on driverless build hosts.
if(MLVC_BACKEND MATCHES "^(driver_cubin|all)$" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(mlvc_c_api_tests PRIVATE
        "LINKER:-rpath-link,${CUDAToolkit_LIBRARY_DIR}/stubs")
endif()
mlvc_configure_target(mlvc_c_api_tests)
mlvc_enable_ipo(mlvc_c_api_tests)

add_test(NAME mlvc_c_api COMMAND mlvc_c_api_tests)

add_executable(mlvc_public_header_tests unit/public_header_tests.c)
target_include_directories(mlvc_public_header_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/public/include")
mlvc_configure_target(mlvc_public_header_tests)
add_test(NAME mlvc_public_header COMMAND mlvc_public_header_tests)

add_executable(mlvc_model_proto_tests unit/model_proto_tests.cpp)
target_include_directories(mlvc_model_proto_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/core/include"
    "${PROJECT_SOURCE_DIR}/runtime/include")
target_link_libraries(mlvc_model_proto_tests PRIVATE
    mlvc_core
    mlvc_runtime
    mlvc_model_proto)
mlvc_configure_target(mlvc_model_proto_tests)
mlvc_enable_ipo(mlvc_model_proto_tests)
add_test(NAME mlvc_model_proto COMMAND mlvc_model_proto_tests)

add_executable(mlvc_runtime_decoupling_tests
    unit/runtime_decoupling_tests.cpp
    $<TARGET_OBJECTS:mlvc_codec_objects>)
target_include_directories(mlvc_runtime_decoupling_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/core/include"
    "${PROJECT_SOURCE_DIR}/runtime/include"
    "${PROJECT_SOURCE_DIR}/codec/include")
target_link_libraries(mlvc_runtime_decoupling_tests PRIVATE
    mlvc_core
    mlvc_runtime
    mlvc_backend
    msrtc_rans
    mlvc_model_proto)
mlvc_configure_target(mlvc_runtime_decoupling_tests)
mlvc_enable_ipo(mlvc_runtime_decoupling_tests)
add_test(NAME mlvc_runtime_decoupling COMMAND mlvc_runtime_decoupling_tests)

if(MLVC_BACKEND MATCHES "^(cpu|all)$")
    add_executable(mlvc_cpu_backend_tests unit/cpu_backend_tests.cpp)
    target_include_directories(mlvc_cpu_backend_tests PRIVATE
        "${PROJECT_SOURCE_DIR}/core/include"
        "${PROJECT_SOURCE_DIR}/runtime/include"
        "${PROJECT_SOURCE_DIR}/codec/include")
    target_link_libraries(mlvc_cpu_backend_tests PRIVATE
        mlvc_core
        mlvc_runtime
        mlvc_backend
        mlvc_model_proto)
    if(UNIX AND NOT APPLE)
        target_link_options(mlvc_cpu_backend_tests PRIVATE "LINKER:--disable-new-dtags")
    endif()
    mlvc_configure_target(mlvc_cpu_backend_tests)
    mlvc_enable_ipo(mlvc_cpu_backend_tests)
    add_test(NAME mlvc_cpu_backend COMMAND mlvc_cpu_backend_tests)
endif()

if(_mlvc_python_protobuf_ok)
    add_test(NAME mlvc_model_package
        COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tests/unit/model_package_tests.py")
endif()

set(_mlvc_conformance_script
    "${PROJECT_SOURCE_DIR}/tests/conformance/run_codec_conformance.py")
if(EXISTS "${MLVC_TEST_ASSETS_DIR}/references/mlvc-psnr-v1/gray-q21-2f/reference.json"
   AND EXISTS "${MLVC_TEST_ASSETS_DIR}/references/mlvc-s-psnr-v1/gray-q21-2f/reference.json")
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME mlvc_codec_reference
        COMMAND mlvc_codec_tests "${MLVC_TEST_ASSETS_DIR}" "${MLVC_MODEL_ROOT}")
    if(NOT MLVC_BACKEND STREQUAL "none")
        foreach(_mlvc_profile mlvc-psnr-v1 mlvc-s-psnr-v1)
            set(_mlvc_model_dir
                "${MLVC_MODEL_ROOT}/${_mlvc_profile}/640x368")
            set(_mlvc_reference_dir
                "${MLVC_TEST_ASSETS_DIR}/references/${_mlvc_profile}/gray-q21-2f")
            if(MLVC_EMBED_MODELS)
                set(_mlvc_conformance_model_arg "embedded:${_mlvc_profile}")
            else()
                set(_mlvc_conformance_model_arg "${_mlvc_model_dir}")
            endif()
            if(MLVC_BACKEND MATCHES "^(driver_cubin|all)$")
                set(_mlvc_conformance_args
                    --binary "$<TARGET_FILE:mlvc_tool_demo>"
                    --backend driver-cubin
                    --model-dir "${_mlvc_conformance_model_arg}"
                    --bundle-dir "${_mlvc_model_dir}"
                    --reference-dir "${_mlvc_reference_dir}")
                add_test(NAME "mlvc_codec_conformance_${_mlvc_profile}"
                    COMMAND "${Python3_EXECUTABLE}" "${_mlvc_conformance_script}"
                            ${_mlvc_conformance_args})
                set_tests_properties("mlvc_codec_conformance_${_mlvc_profile}"
                    PROPERTIES LABELS "gpu;conformance" TIMEOUT 300)
            endif()
            if(MLVC_BACKEND MATCHES "^(cpu|all)$")
                set(_mlvc_cpu_conformance_args
                    --binary "$<TARGET_FILE:mlvc_tool_demo>"
                    --backend cpu
                    --model-dir "${_mlvc_conformance_model_arg}"
                    --bundle-dir "${_mlvc_model_dir}"
                    --reference-dir "${_mlvc_reference_dir}")
                if(CMAKE_CROSSCOMPILING_EMULATOR)
                    list(APPEND _mlvc_cpu_conformance_args
                        --emulator "${CMAKE_CROSSCOMPILING_EMULATOR}")
                endif()
                if(MLVC_BACKEND STREQUAL "all")
                    set(_mlvc_cpu_test_name "mlvc_codec_conformance_cpu_${_mlvc_profile}")
                else()
                    set(_mlvc_cpu_test_name "mlvc_codec_conformance_${_mlvc_profile}")
                endif()
                add_test(NAME "${_mlvc_cpu_test_name}"
                    COMMAND "${Python3_EXECUTABLE}" "${_mlvc_conformance_script}"
                            ${_mlvc_cpu_conformance_args})
                set_tests_properties("${_mlvc_cpu_test_name}"
                    PROPERTIES LABELS "cpu;conformance" TIMEOUT 600)
            endif()
        endforeach()
        add_custom_target(mlvc_codec_compatibility
            COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L conformance
            DEPENDS mlvc_tool_demo
            USES_TERMINAL
            COMMENT "Checking both MLVC profiles against official Python codec references")
    else()
        add_custom_target(mlvc_codec_compatibility
            COMMAND "${CMAKE_COMMAND}" -E echo
                "Codec conformance requires an active inference backend (MLVC_BACKEND=${MLVC_BACKEND})"
            USES_TERMINAL
            COMMENT "Skipping end-to-end inference conformance in backend-none builds")
    endif()
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
