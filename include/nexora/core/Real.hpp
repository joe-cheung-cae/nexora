#pragma once
#include <nexora/core/Config.hpp>

namespace nexora {

// Selected when Nexora is configured; installed consumers inherit the same type.
#if NEXORA_ENABLE_DOUBLE_PRECISION
using real = double;
#else
using real = float;
#endif

} // namespace nexora
