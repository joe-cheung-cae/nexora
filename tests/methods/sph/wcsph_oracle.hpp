#pragma once
#include "cell_grid_oracle.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <methods/sph/Pressure3D.hpp>
namespace sph_test {
struct ForceOracle {
    Oracle csr;
    std::map<std::uint64_t, long double> pressure, pressure_scale;
    std::map<std::uint64_t, std::array<long double, 3>> acceleration, acceleration_scale;
};
inline ForceOracle force_oracle(const std::vector<Particle>& input, sph::WcsphParameters parameters,
                                const std::map<std::uint64_t, long double>& density = {}) {
    auto all_local = input;
    for (auto& particle : all_local)
        particle.ghost = 0;
    ForceOracle result;
    result.csr = explicit_csr(all_local); // Ghost density needs its complete local CSR too.
    if (!density.empty())
        result.csr.density = density;
    const long double pi = std::acos(-1.0L);
    const long double rho0 = parameters.reference_density, c = parameters.sound_speed;
    const long double gamma = parameters.gamma, prefactor = rho0 * c * c / gamma;
    std::map<std::uint64_t, const Particle*> particles;
    for (const auto& particle : input) {
        particles.emplace(particle.id, &particle);
        const real rho = static_cast<real>(result.csr.density.at(particle.id));
        const long double power = std::pow(static_cast<long double>(rho) / rho0, gamma);
        result.pressure[particle.id] = static_cast<real>(prefactor * (power - 1));
        result.pressure_scale[particle.id] = prefactor * (power + 1);
    }
    for (const auto& particle : input) {
        if (particle.ghost)
            continue;
        auto& acceleration = result.acceleration[particle.id];
        auto& scale = result.acceleration_scale[particle.id];
        const long double rho_i = static_cast<real>(result.csr.density.at(particle.id));
        const auto row = result.csr.row_by_id.at(particle.id);
        for (auto edge = result.csr.offsets[row]; edge < result.csr.offsets[row + 1]; ++edge) {
            const auto& neighbor = *particles.at(result.csr.ids[edge]);
            double d2 = 0;
            std::array<double, 3> delta;
            for (int k = 0; k < 3; ++k) {
                delta[k] = double(particle.position[k]) - double(neighbor.position[k]);
                d2 = std::fma(delta[k], delta[k], d2);
            }
            const long double q = std::sqrt(static_cast<long double>(d2));
            // Differentiate the oracle's truncated powers, expanding the inner piece
            // to retain the finite radial factor at zero and at small separations.
            const long double outer = std::max(2 - q, 0.0L);
            const long double radial =
                q < 1 ? (9 * q - 12) / (4 * pi) : -3 * outer * outer / (4 * pi * q);
            const long double rho_j = static_cast<real>(result.csr.density.at(neighbor.id));
            const long double coefficient =
                neighbor.mass * (result.pressure.at(particle.id) / (rho_i * rho_i) +
                                 result.pressure.at(neighbor.id) / (rho_j * rho_j));
            const long double coefficient_scale =
                neighbor.mass * (result.pressure_scale.at(particle.id) / (rho_i * rho_i) +
                                 result.pressure_scale.at(neighbor.id) / (rho_j * rho_j));
            for (int k = 0; k < 3; ++k) {
                const long double gradient = radial * delta[k];
                acceleration[k] -= coefficient * gradient;
                scale[k] += coefficient_scale * std::abs(gradient);
            }
        }
    }
    return result;
}
// Both participant snapshots and direct operator results expose these observations.
// Keeping one checker preserves the independent CSR reference and error bounds.
template <class Observation>
inline void check_force_snapshot(const Observation& snapshot,
                                 const std::vector<Particle>& input,
                                 sph::WcsphParameters parameters,
                                 const std::map<std::uint64_t, long double>& density = {}) {
    const auto oracle = force_oracle(input, parameters, density);
    ASSERT_EQ(snapshot.row_count(), oracle.acceleration.size());
    std::vector<std::uint64_t> seen;
    for (std::size_t row = 0; row < snapshot.row_count(); ++row) {
        const auto id = snapshot.row_id(row);
        seen.push_back(id);
        ASSERT_TRUE(oracle.acceleration.contains(id)); // No ghost output row.
        const auto csr_row = oracle.csr.row_by_id.at(id);
        const std::vector<std::uint64_t> expected(
            oracle.csr.ids.begin() + oracle.csr.offsets[csr_row],
            oracle.csr.ids.begin() + oracle.csr.offsets[csr_row + 1]);
        std::vector<std::uint64_t> actual;
        snapshot.for_each_neighbor(row, [&](std::uint64_t other, double d2) {
            EXPECT_NE(other, id);
            EXPECT_LE(d2, 4);
            actual.push_back(other);
        });
        std::sort(actual.begin(), actual.end());
        EXPECT_EQ(actual, expected);
        const long double epsilon = std::numeric_limits<real>::epsilon();
        EXPECT_LE(
            std::abs(static_cast<long double>(snapshot.density(row)) - oracle.csr.density.at(id)),
            32 * (expected.size() + 1) * epsilon *
                oracle.csr.density.at(id));
        const auto pressure = checked(snapshot.pressure(row));
        EXPECT_LE(std::abs(static_cast<long double>(pressure) - oracle.pressure.at(id)),
                  64 * epsilon * oracle.pressure_scale.at(id));
        const auto acceleration = checked(snapshot.pressure_acceleration(row));
        for (int k = 0; k < 3; ++k) {
            EXPECT_TRUE(std::isfinite(acceleration[k]));
            EXPECT_LE(
                std::abs(static_cast<long double>(acceleration[k]) - oracle.acceleration.at(id)[k]),
                256 * epsilon * oracle.acceleration_scale.at(id)[k])
                << "ID=" << id << " axis=" << k;
        }
    }
    std::sort(seen.begin(), seen.end());
    EXPECT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end());
}
} // namespace sph_test
