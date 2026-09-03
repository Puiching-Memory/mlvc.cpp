set(MLVC_DRIVER_FATBIN
    "${CMAKE_CURRENT_SOURCE_DIR}/assets/cubin/mlvc_driver_kernels.fatbin")
if(NOT EXISTS "${MLVC_DRIVER_FATBIN}")
    message(FATAL_ERROR
        "Missing prebuilt driver fatbin: ${MLVC_DRIVER_FATBIN}. "
        "Run scripts/build_driver_fatbin.sh on a CUDA build host.")
endif()

# Official CUDA headers and the link stub are build inputs. The deployed
# executable requires the NVIDIA driver but no CUDA Toolkit runtime libraries.
find_package(CUDAToolkit 13.3 REQUIRED)

set(MLVC_EMBEDDED_FATBIN_SOURCE
    "${CMAKE_CURRENT_BINARY_DIR}/generated/mlvc_driver_kernels_fatbin.cpp")
add_custom_command(
    OUTPUT "${MLVC_EMBEDDED_FATBIN_SOURCE}"
    COMMAND "${CMAKE_COMMAND}"
        "-DINPUT=${MLVC_DRIVER_FATBIN}"
        "-DOUTPUT=${MLVC_EMBEDDED_FATBIN_SOURCE}"
        -DSYMBOL=mlvc_driver_kernels_fatbin
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
    DEPENDS
        "${MLVC_DRIVER_FATBIN}"
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
    VERBATIM)

add_library(mlvc_driver_kernels STATIC
    "${MLVC_EMBEDDED_FATBIN_SOURCE}"
)
set_target_properties(mlvc_driver_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

add_library(mlvc_driver STATIC
    src/driver/driver.cpp
)
set_target_properties(mlvc_driver PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(mlvc_driver PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CUDAToolkit_INCLUDE_DIRS}"
)
target_link_libraries(mlvc_driver PUBLIC CUDA::cuda_driver)

add_executable(mlvc_driver_probe
    src/driver_probe.cpp
)
target_link_libraries(mlvc_driver_probe PRIVATE
    mlvc_driver mlvc_driver_kernels)
target_link_options(mlvc_driver_probe PRIVATE "LINKER:--disable-new-dtags")

if(MLVC_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(LANGUAGES CXX
        RESULT MLVC_DRIVER_IPO_SUPPORTED OUTPUT MLVC_DRIVER_IPO_ERROR)
    if(MLVC_DRIVER_IPO_SUPPORTED)
        set_property(TARGET mlvc_driver mlvc_driver_kernels mlvc_driver_probe PROPERTY
            INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    else()
        message(WARNING "Driver-cubin IPO is unavailable: ${MLVC_DRIVER_IPO_ERROR}")
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME mlvc_driver_cubin_probe
             COMMAND mlvc_driver_probe --iterations 100)
endif()

install(TARGETS mlvc_driver_probe RUNTIME DESTINATION bin)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
        DESTINATION "share/licenses/mlvc_cpp")
