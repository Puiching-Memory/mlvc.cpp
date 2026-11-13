#include "mlvc/runtime/backend.hpp"

#ifdef MLVC_HAS_BACKEND_DRIVER_CUBIN
namespace mlvc::driver_cubin_backend {
void register_driver_cubin_backend();
}
#endif

#ifdef MLVC_HAS_BACKEND_CPU
namespace mlvc::cpu_backend {
void register_cpu_backend(bool make_default);
}
#endif

namespace mlvc::detail {

extern bool g_selected_backends_registered;
bool g_selected_backends_registered = []() {
#ifdef MLVC_HAS_BACKEND_DRIVER_CUBIN
    driver_cubin_backend::register_driver_cubin_backend();
#endif
#ifdef MLVC_HAS_BACKEND_CPU
#ifdef MLVC_HAS_BACKEND_DRIVER_CUBIN
    cpu_backend::register_cpu_backend(false);
#else
    cpu_backend::register_cpu_backend(true);
#endif
#endif
    return true;
}();

}  // namespace mlvc::detail
