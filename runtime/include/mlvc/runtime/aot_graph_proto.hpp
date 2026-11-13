#pragma once

#include "mlvc/runtime/aot_graph.hpp"
#include "mlvc_model.pb.h"

namespace mlvc::aot {

GraphDescriptor decode_graph(const mlvc::wire::Graph& source);

}  // namespace mlvc::aot
