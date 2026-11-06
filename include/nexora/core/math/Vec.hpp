#pragma once
#include <array>
#include <cstddef>
namespace nexora {
template <class T, std::size_t N> struct Vec {
    std::array<T, N> values{};
    T& operator[](std::size_t i) noexcept {
        return values[i];
    }
    const T& operator[](std::size_t i) const noexcept {
        return values[i];
    }
};
} // namespace nexora
