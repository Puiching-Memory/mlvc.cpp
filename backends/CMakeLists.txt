add_library(mlvc_backend_registration OBJECT selected_backends.cpp)
set_target_properties(mlvc_backend_registration PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(mlvc_backend_registration PRIVATE mlvc_runtime)
mlvc_configure_target(mlvc_backend_registration)
mlvc_enable_ipo(mlvc_backend_registration)

add_library(mlvc_backend INTERFACE)
target_sources(mlvc_backend INTERFACE $<TARGET_OBJECTS:mlvc_backend_registration>)
target_link_libraries(mlvc_backend INTERFACE mlvc_runtime)

if(MLVC_BACKEND MATCHES "^(driver_cubin|all)$")
    add_subdirectory(driver_cubin)
    target_compile_definitions(mlvc_backend_registration PRIVATE
        MLVC_HAS_BACKEND_DRIVER_CUBIN=1)
    target_link_libraries(mlvc_backend INTERFACE mlvc_backend_driver_cubin)
endif()

if(MLVC_BACKEND MATCHES "^(cpu|all)$")
    add_subdirectory(cpu)
    target_compile_definitions(mlvc_backend_registration PRIVATE
        MLVC_HAS_BACKEND_CPU=1)
    target_link_libraries(mlvc_backend INTERFACE mlvc_backend_cpu)
endif()
