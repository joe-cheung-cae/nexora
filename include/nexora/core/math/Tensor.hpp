#pragma once
#include <array>
#include <cstddef>
namespace nexora {
template <class T, std::size_t M, std::size_t N> struct Tensor {
    std::array<T, M * N> values{};
    T& operator()(std::size_t i, std::size_t j) noexcept {
        return values[i * N + j];
    }
    const T& operator()(std::size_t i, std::size_t j) const noexcept {
        return values[i * N + j];
    }
};
} // namespace nexora
