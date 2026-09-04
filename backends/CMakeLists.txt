add_subdirectory("${MLVC_SELECTED_BACKEND}")
add_library(mlvc_backend_selected ALIAS
    "mlvc_backend_${MLVC_SELECTED_BACKEND}")
