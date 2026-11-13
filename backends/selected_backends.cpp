#include "mlvc/runtime/backend.hpp"

#ifdef MLVC_HAS_BACKEND_DRIVER_CUBIN
namespace mlvc::driver_cubin_backend {
void register_driver_cubin_backend();
}
#endif

namespace mlvc {
namespace {

[[maybe_unused]] const bool g_mlvc_backends_registered = []() {
#ifdef MLVC_HAS_BACKEND_DRIVER_CUBIN
    driver_cubin_backend::register_driver_cubin_backend();
#endif
    return true;
}();

}  // namespace
}  // namespace mlvc
