set(MLVC_CORE_SOURCES
    bit_io.cpp
    bitstream.cpp
    entropy.cpp
    logging.cpp
    model.cpp
    model_assets.cpp
    scales.cpp
    sha256.cpp
    yuv.cpp)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _mlvc_core_arch)
set(_mlvc_core_msvc_x64 FALSE)
if(MSVC AND _mlvc_core_arch MATCHES "^(x86_64|amd64)$")
    set(_mlvc_core_msvc_x64 TRUE)
    set_source_files_properties(entropy_x86_avx2.cpp PROPERTIES
        COMPILE_OPTIONS "/arch:AVX2")
    list(APPEND MLVC_CORE_SOURCES entropy_x86_avx2.cpp)
endif()

add_library(mlvc_core STATIC ${MLVC_CORE_SOURCES})
if(_mlvc_core_msvc_x64)
    target_compile_definitions(mlvc_core PRIVATE MLVC_CORE_HAS_MSVC_X86_AVX2=1)
endif()
set_target_properties(mlvc_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(mlvc_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(mlvc_core PRIVATE
    msrtc_rans
    mlvc_model_proto)
mlvc_configure_target(mlvc_core)
mlvc_enable_ipo(mlvc_core)

if(MLVC_EMBED_MODELS)
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux|Android|OHOS" OR MSVC)
        message(FATAL_ERROR
            "MLVC_EMBED_MODELS=ON currently requires an ELF target with GNU/Clang ASM (.incbin)")
    endif()
    file(GLOB MLVC_MODEL_MANIFESTS CONFIGURE_DEPENDS
        LIST_DIRECTORIES false "${MLVC_MODEL_ROOT}/*/*/model_bundle.pb")
    if(NOT MLVC_MODEL_MANIFESTS)
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
    set(MLVC_EMBEDDED_MODELS_DEPFILE
        "${MLVC_EMBEDDED_MODELS_DIR}/mlvc_embedded_models.d")
    file(MAKE_DIRECTORY "${MLVC_EMBEDDED_MODELS_DIR}")
    configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/EmbedModelData.S.in"
        "${MLVC_EMBEDDED_MODELS_ASSEMBLY}" @ONLY)

    add_custom_command(
        OUTPUT
            "${MLVC_EMBEDDED_MODELS_BINARY}"
            "${MLVC_EMBEDDED_MODELS_INDEX}"
        COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/tools/python/mlvc_tools/package.py" embed
            --model-root "${MLVC_MODEL_ROOT}"
            --output-data "${MLVC_EMBEDDED_MODELS_BINARY}"
            --output-index "${MLVC_EMBEDDED_MODELS_INDEX}"
            --depfile "${MLVC_EMBEDDED_MODELS_DEPFILE}"
        DEPENDS
            ${MLVC_MODEL_MANIFESTS}
            "${MLVC_PROFILE_FILE}"
            "${PROJECT_SOURCE_DIR}/tools/python/mlvc_tools/__init__.py"
            "${PROJECT_SOURCE_DIR}/tools/python/mlvc_tools/package.py"
            "${PROJECT_SOURCE_DIR}/tools/python/mlvc_tools/wire.py"
            "${PROJECT_SOURCE_DIR}/tools/python/mlvc_tools/proto/mlvc_model_pb2.py"
        DEPFILE "${MLVC_EMBEDDED_MODELS_DEPFILE}"
        COMMENT "Embedding validated driver-cubin model bundles"
        VERBATIM)

    set_source_files_properties("${MLVC_EMBEDDED_MODELS_INDEX}" PROPERTIES
        GENERATED TRUE
        HEADER_FILE_ONLY TRUE)
    set_source_files_properties("${MLVC_EMBEDDED_MODELS_ASSEMBLY}" PROPERTIES
        OBJECT_DEPENDS "${MLVC_EMBEDDED_MODELS_BINARY}")
    set_source_files_properties(model_assets.cpp PROPERTIES
        OBJECT_DEPENDS "${MLVC_EMBEDDED_MODELS_INDEX}")
    target_sources(mlvc_core PRIVATE
        "${MLVC_EMBEDDED_MODELS_INDEX}"
        "${MLVC_EMBEDDED_MODELS_ASSEMBLY}")
    target_include_directories(mlvc_core PRIVATE "${MLVC_EMBEDDED_MODELS_DIR}")
    target_compile_definitions(mlvc_core PRIVATE MLVC_HAS_EMBEDDED_MODELS=1)
endif()

install(FILES "${PROJECT_SOURCE_DIR}/third_party/libmlvc/LICENSE"
    DESTINATION "share/licenses/libmlvc")
