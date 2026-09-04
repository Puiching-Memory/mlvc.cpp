set(MLVC_CORE_SOURCES
    src/bitstream.cpp
    src/entropy.cpp
    src/model.cpp
    src/model_assets.cpp
    src/scales.cpp
    src/yuv.cpp)

add_library(mlvc_core STATIC ${MLVC_CORE_SOURCES})
set_target_properties(mlvc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${NLOHMANN_JSON_DIR}/single_include)
target_link_libraries(mlvc_core PRIVATE
    msrtc_rans
    nlohmann_json::nlohmann_json)
mlvc_enable_ipo(mlvc_core)

if(MLVC_SELECTED_BACKEND STREQUAL "driver_cubin")
    file(GLOB_RECURSE MLVC_EMBEDDED_MODEL_INPUTS CONFIGURE_DEPENDS
        LIST_DIRECTORIES false "${MLVC_MODEL_ROOT}/*")
    if(NOT MLVC_EMBEDDED_MODEL_INPUTS)
        message(FATAL_ERROR
            "No canonical model bundles found below ${MLVC_MODEL_ROOT}")
    endif()

    set(MLVC_EMBEDDED_MODELS_DIR
        "${PROJECT_BINARY_DIR}/generated/embedded-models")
    set(MLVC_EMBEDDED_MODELS_BINARY
        "${MLVC_EMBEDDED_MODELS_DIR}/mlvc_driver_models.bin")
    set(MLVC_EMBEDDED_MODELS_INDEX
        "${MLVC_EMBEDDED_MODELS_DIR}/mlvc_embedded_models_index.inc")
    set(MLVC_EMBEDDED_MODELS_ASSEMBLY
        "${MLVC_EMBEDDED_MODELS_DIR}/mlvc_embedded_models.S")
    file(MAKE_DIRECTORY "${MLVC_EMBEDDED_MODELS_DIR}")
    configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/EmbedModelData.S.in"
        "${MLVC_EMBEDDED_MODELS_ASSEMBLY}" @ONLY)

    add_custom_command(
        OUTPUT
            "${MLVC_EMBEDDED_MODELS_BINARY}"
            "${MLVC_EMBEDDED_MODELS_INDEX}"
        COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tools/model_package.py" embed
            --model-root "${MLVC_MODEL_ROOT}"
            --output-data "${MLVC_EMBEDDED_MODELS_BINARY}"
            --output-index "${MLVC_EMBEDDED_MODELS_INDEX}"
        DEPENDS
            ${MLVC_EMBEDDED_MODEL_INPUTS}
            "${MLVC_PROFILE_FILE}"
            "${PROJECT_SOURCE_DIR}/tools/model_package.py"
        COMMENT "Embedding validated driver-cubin model bundles"
        VERBATIM)

    set_source_files_properties("${MLVC_EMBEDDED_MODELS_INDEX}" PROPERTIES
        GENERATED TRUE
        HEADER_FILE_ONLY TRUE)
    set_source_files_properties("${MLVC_EMBEDDED_MODELS_ASSEMBLY}" PROPERTIES
        OBJECT_DEPENDS "${MLVC_EMBEDDED_MODELS_BINARY}")
    set_source_files_properties(src/model_assets.cpp PROPERTIES
        OBJECT_DEPENDS "${MLVC_EMBEDDED_MODELS_INDEX}")
    target_sources(mlvc_core PRIVATE
        "${MLVC_EMBEDDED_MODELS_INDEX}"
        "${MLVC_EMBEDDED_MODELS_ASSEMBLY}")
    target_include_directories(mlvc_core PRIVATE "${MLVC_EMBEDDED_MODELS_DIR}")
    target_compile_definitions(mlvc_core PRIVATE MLVC_HAS_EMBEDDED_MODELS=1)
endif()
