add_executable(mlvc_webgpu_wasm mlvc_webgpu_wasm.cpp)

target_include_directories(mlvc_webgpu_wasm PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(mlvc_webgpu_wasm PRIVATE
    mlvc_core
    mlvc_runtime
    mlvc_backend
    mlvc_backend_webgpu
    msrtc_rans
    mlvc_model_proto
)

set_target_properties(mlvc_webgpu_wasm PROPERTIES
    OUTPUT_NAME "mlvc_webgpu"
    SUFFIX ".js"
)

target_compile_options(mlvc_webgpu_wasm PRIVATE
    -O3
)

target_link_options(mlvc_webgpu_wasm PRIVATE
    -O3
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sENVIRONMENT=web,worker
    -sALLOW_MEMORY_GROWTH=1
    -sINITIAL_MEMORY=67108864
    -sMAXIMUM_MEMORY=1073741824
    -sNO_EXIT_RUNTIME=1
    -sEXPORTED_FUNCTIONS=_malloc,_free,_mlvc_web_init_gpu,_mlvc_web_flush_gpu,_mlvc_web_asset_clear,_mlvc_web_asset_put,_mlvc_web_decoder_create,_mlvc_web_decoder_reset,_mlvc_web_decoder_destroy,_mlvc_web_decode_au,_mlvc_web_last_error,_mlvc_web_gpu_info
    -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,HEAPU8,HEAPF64,HEAP32,UTF8ToString,stringToUTF8,lengthBytesUTF8
)

mlvc_configure_target(mlvc_webgpu_wasm)
