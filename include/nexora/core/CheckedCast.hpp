#pragma once
#include <cstdint>
#include <limits>
#include <nexora/core/Result.hpp>
#include <type_traits>
namespace nexora {
template <class To, class From> Result<To> checked_integer_cast(From value) {
    static_assert(std::is_integral<To>::value && std::is_integral<From>::value,
                  "integer types required");
    static_assert(!std::is_same<To, bool>::value && !std::is_same<From, bool>::value,
                  "bool is not supported");
    if constexpr (std::is_signed<From>::value) {
        if (value < 0) {
            if constexpr (!std::is_signed<To>::value)
                return Status{StatusCode::invalid_argument, "integer conversion out of range"};
            else if (static_cast<std::intmax_t>(value) <
                     static_cast<std::intmax_t>(std::numeric_limits<To>::min()))
                return Status{StatusCode::invalid_argument, "integer conversion out of range"};
        } else if (static_cast<std::uintmax_t>(value) >
                   static_cast<std::uintmax_t>(std::numeric_limits<To>::max()))
            return Status{StatusCode::invalid_argument, "integer conversion out of range"};
    } else if (static_cast<std::uintmax_t>(value) >
               static_cast<std::uintmax_t>(std::numeric_limits<To>::max()))
        return Status{StatusCode::invalid_argument, "integer conversion out of range"};
    return static_cast<To>(value);
}
} // namespace nexora
