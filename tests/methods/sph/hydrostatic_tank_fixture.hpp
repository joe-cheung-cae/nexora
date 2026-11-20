#pragma once

#include "cell_grid_oracle.hpp"

#include <cmath>
#include <methods/sph/Pressure3D.hpp>

namespace sph_test::hydrostatic {
inline constexpr double tank_size = 6, gravity = 9.81;
inline constexpr sph::WcsphParameters eos{1000, 80, 7};

inline double depth_term(double z) {
    return (eos.gamma - 1) * gravity * (tank_size - z) /
           (eos.sound_speed * eos.sound_speed);
}

inline double log_density_ratio(double z) {
    return std::log1p(depth_term(z)) / (eos.gamma - 1);
}

inline double hydrostatic_density(double z) {
    return eos.reference_density * std::pow(1 + depth_term(z), 1 / (eos.gamma - 1));
}

inline real conventional_mass(double z, double spacing) {
    return static_cast<real>(hydrostatic_density(z) * spacing * spacing * spacing);
}

struct Tank {
    std::vector<Particle> fluid, wall;
};

inline int wall_layers(double h, double wall_spacing) {
    return static_cast<int>(std::ceil(2 * h / wall_spacing));
}

inline Tank make_tank(int n, double h, std::uint64_t first_wall_id = 0) {
    const double dx = tank_size / n;
    const int layers = wall_layers(h, dx);
    if (!first_wall_id)
        first_wall_id = static_cast<std::uint64_t>(n) * n * n + 1;
    Tank tank;
    for (int z = -layers; z < n; ++z)
        for (int y = -layers; y < n + layers; ++y)
            for (int x = -layers; x < n + layers; ++x) {
                const bool fluid = z >= 0 && x >= 0 && x < n && y >= 0 && y < n;
                auto& particles = fluid ? tank.fluid : tank.wall;
                const std::uint64_t id = particles.size() + (fluid ? 1 : first_wall_id);
                const double center_z = (z + .5) * dx;
                particles.push_back({{static_cast<real>((x + .5) * dx),
                                      static_cast<real>((y + .5) * dx),
                                      static_cast<real>(center_z)},
                                     conventional_mass(center_z, dx), id, 0});
            }
    return tank;
}
} // namespace sph_test::hydrostatic
