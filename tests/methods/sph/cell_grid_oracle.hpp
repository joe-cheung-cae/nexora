#pragma once
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <methods/sph/neighbors/CellGrid.hpp>
#include <stdexcept>
#include <utility>
#include <vector>
namespace sph_test {
using namespace nexora;
using Position = Kokkos::View<real* [3], Kokkos::LayoutLeft, Kokkos::HostSpace>;
template <class T> using Array = Kokkos::View<T*, Kokkos::HostSpace>;
struct Particle {
    std::array<real, 3> position;
    real mass;
    std::uint64_t id;
    std::uint8_t ghost;
};
struct Fields {
    Position position;
    Array<real> mass;
    Array<std::uint64_t> ids;
    Array<std::uint8_t> ghost;
    explicit Fields(const std::vector<Particle>& input)
        : position("test.position", input.size()), mass("test.mass", input.size()),
          ids("test.ids", input.size()), ghost("test.ghost", input.size()) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            for (int k = 0; k < 3; ++k)
                position(i, k) = input[i].position[k];
            mass(i) = input[i].mass;
            ids(i) = input[i].id;
            ghost(i) = input[i].ghost;
        }
    }
    sph::ParticleFields3D get() const {
        return {position, mass, ids, ghost};
    }
};
template <class T> T checked(Result<T> result) {
    if (!result.ok())
        throw std::runtime_error(result.status().message());
    return std::move(result).value();
}

// This oracle uses only source particles and an all-pairs scan, never cell keys,
// packed slot ranges, or the production neighbor enumerator to construct CSR.
// Membership requires exact component distances <=2 followed by the public
// binary64 squared-distance <=4 contract. Diagonal classification is not exact-real.
struct Oracle {
    std::vector<std::size_t> offsets{0};
    std::vector<std::uint64_t> ids;
    std::map<std::uint64_t, long double> density;
    std::map<std::uint64_t, std::size_t> row_by_id;
};
// Independent error-free TwoDiff: the rounded difference plus residual is
// the exact component distance of the two binary64 source coordinates.
// Ordinary cutoff neighbors such as nextafter(2, 0) and 4 can subtract to 2
// after rounding even though their exact distance exceeds the support radius.
inline bool exact_component_support(real first, real second) {
    const double high = std::max(double(first), double(second));
    const double low = std::min(double(first), double(second));
    const double difference = high - low;
    if (difference != 2.0)
        return difference < 2.0;
    const double virtual_low = high - difference;
    const double virtual_high = difference + virtual_low;
    const double low_roundoff = virtual_low - low;
    const double high_roundoff = high - virtual_high;
    const double residual = high_roundoff + low_roundoff;
    return residual <= 0;
}
inline Oracle explicit_csr(const std::vector<Particle>& input) {
    Oracle oracle;
    std::map<std::uint64_t, const Particle*> by_id;
    for (const auto& particle : input)
        by_id.emplace(particle.id, &particle);
    const auto distance_squared = [](const Particle& a, const Particle& b) {
        double sum = 0;
        for (int k = 0; k < 3; ++k) {
            const double d = double(a.position[k]) - double(b.position[k]);
            sum = std::fma(d, d, sum);
        }
        return sum;
    };
    const long double pi = std::acos(-1.0L);
    for (const auto& owned : input) {
        if (owned.ghost)
            continue;
        oracle.row_by_id.emplace(owned.id, oracle.offsets.size() - 1);
        std::vector<std::uint64_t> neighbors;
        for (const auto& source : input) {
            bool component_support = true;
            for (int axis = 0; axis < 3; ++axis)
                component_support &=
                    exact_component_support(owned.position[axis], source.position[axis]);
            if (source.id != owned.id && component_support && distance_squared(owned, source) <= 4)
                neighbors.push_back(source.id);
        }
        std::sort(neighbors.begin(), neighbors.end());
        oracle.ids.insert(oracle.ids.end(), neighbors.begin(), neighbors.end());
        oracle.offsets.push_back(oracle.ids.size());
        // Evaluate from the explicit CSR row, adding self once separately.
        long double density = static_cast<long double>(owned.mass) / pi;
        for (std::size_t edge = oracle.offsets[oracle.offsets.size() - 2];
             edge < oracle.offsets.back(); ++edge) {
            const auto& source = *by_id.at(oracle.ids[edge]);
            const long double q =
                std::sqrt(static_cast<long double>(distance_squared(owned, source)));
            // Independent truncated-power cubic, equivalent to the 3D M4 spline.
            const auto cube = [](long double v) { return v * v * v; };
            const long double weight =
                (cube(std::max(2 - q, 0.0L)) - 4 * cube(std::max(1 - q, 0.0L))) / (4 * pi);
            density += static_cast<long double>(source.mass) * weight;
        }
        oracle.density.emplace(owned.id, density);
    }
    return oracle;
}
} // namespace sph_test
