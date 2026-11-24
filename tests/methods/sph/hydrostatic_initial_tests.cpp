#include "hydrostatic_tank_fixture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <locale>
#include <methods/sph/Density3D.hpp>
#include <nexora/simulation/Simulation.hpp>
#include <numbers>
#include <participants/sph/Participant3D.hpp>
#include <vector>

namespace {
using namespace nexora;
using namespace sph_test;
using namespace sph_test::hydrostatic;

constexpr double tait_b = eos.reference_density * eos.sound_speed * eos.sound_speed / eos.gamma;
constexpr double field_tolerance = 4 * std::numeric_limits<real>::epsilon();

struct TankCase {
    const char* name;
    int n;
    double h;
    int layers;
    double pressure_rms_pa;
    double acceleration_residual_rms_over_g;
};

// The coarse reference is from the original conventional tank. The three
// finer cases use the measured Release double initial fields as regressions.
constexpr TankCase cases[]{
    {"n6_h1", 6, 1, 2, 253971, 25.698},
    {"n12_h0p5", 12, .5, 2, 179150, 36.348},
    {"n24_h0p25", 24, .25, 2, 126525, 51.405},
    {"n24_h0p5", 24, .5, 4, 193663, 68.349},
};

struct Moments {
    double partition{};
    std::array<double, 3> gradient{};
    double density{};
    std::size_t count{};
};

// The filled solid shell retains every integer offset within support except
// those extending above the open surface. An independent truncated-power
// spline over these offsets tests the cell traversal and physical kernel scales.
std::vector<Moments> reference_layers(int n, int layers, double dx, double h) {
    const double volume = dx * dx * dx;
    const double normalization = 1 / (4 * std::numbers::pi * h * h * h);
    const auto square = [](double x) { return x * x; };
    std::vector<Moments> result(n);
    for (int z = 0; z < n; ++z) {
        auto& reference = result[z];
        for (int k = -layers; k <= layers && z + k < n; ++k)
            for (int j = -layers; j <= layers; ++j)
                for (int i = -layers; i <= layers; ++i) {
                    const int r2 = i * i + j * j + k * k;
                    const double q = std::sqrt(r2) * dx / h;
                    if (q > 2)
                        continue;
                    const double inner = std::max(1 - q, 0.);
                    const double outer = std::max(2 - q, 0.);
                    const double weight = normalization *
                                          (square(outer) * outer - 4 * square(inner) * inner);
                    reference.partition += volume * weight;
                    reference.density += conventional_mass((z + k + .5) * dx, dx) * weight;
                    if (r2 > 0)
                        reference.gradient[2] += volume * square(k * dx) * normalization *
                                                 (3 * square(outer) - 12 * square(inner)) /
                                                 (q * h * h);
                    ++reference.count;
                }
    }
    return result;
}

void measure(TankCase fixture, std::ostream* csv) {
    SCOPED_TRACE(fixture.name);
    const double dx = tank_size / fixture.n;
    const double volume = dx * dx * dx;
    const double h_squared = fixture.h * fixture.h;
    const double h_cubed = h_squared * fixture.h;
    const int layers = wall_layers(fixture.h, dx);
    const auto tank = make_tank(fixture.n, fixture.h);
    const Fields fluid(tank.fluid), wall(tank.wall);
    const sph::Velocity3D rest("initial_tank.rest", tank.fluid.size());
    const sph::WcsphVerletParameters parameters{eos, .5, {0, 0, -gravity}, fixture.h};
    const sph::FixedWallParticles3D fixed{wall.position, wall.mass, wall.ids};
    const auto reference = reference_layers(fixture.n, layers, dx, fixture.h);
    sph::Participant3D participant(fluid.get(), 10000, parameters, rest, fixed);
    Simulation simulation(participant);
    const auto initialized = simulation.initialize();
    ASSERT_TRUE(initialized.ok()) << initialized.message();
    const auto prepared = simulation.prepare();
    ASSERT_TRUE(prepared.ok()) << prepared.message();
    const auto state = checked(participant.snapshot());
    ASSERT_EQ(state.row_count(), tank.fluid.size());
    EXPECT_EQ(simulation.time().accepted_steps, 0);

    // IDs are consecutive within each source array, independent of packed cell order.
    const auto source_by_id = [&](std::uint64_t id) -> const Particle& {
        if (id <= tank.fluid.size())
            return tank.fluid.at(id - 1);
        return tank.wall.at(id - tank.fluid.size() - 1);
    };
    std::vector<bool> seen(tank.fluid.size());
    double total_mass = 0;
    double pressure_error_sum = 0;
    double acceleration_error_sum = 0;
    for (std::size_t row = 0; row < state.row_count(); ++row) {
        const auto id = state.row_id(row);
        SCOPED_TRACE(id);
        ASSERT_GT(id, 0);
        ASSERT_LE(id, tank.fluid.size());
        ASSERT_FALSE(seen[id - 1]);
        seen[id - 1] = true;
        const auto position = checked(state.position(row));
        const auto& center = source_by_id(id);
        EXPECT_EQ(position, center.position);
        const double density = state.density(row);
        const double pressure = checked(state.pressure(row));
        const auto ap = checked(state.pressure_acceleration(row));

        const double self_weight = sph::m4_weight_3d(0) / h_cubed;
        Moments observed{volume * self_weight, {}, center.mass * self_weight, 1};
        // Counts include self, wall sources and admitted q=2 neighbors, whose
        // spline weight and gradient are zero.
        state.for_each_neighbor(row, [&](std::uint64_t source_id, double d2) {
            const auto& source = source_by_id(source_id);
            const double q = std::sqrt(d2) / fixture.h;
            const double w = sph::m4_weight_3d(q) / h_cubed;
            const double dz = double(source.position[2]) - position[2];
            const double gradient = sph::m4_negative_radial_derivative_over_q_3d(q) /
                                    (h_cubed * h_squared);
            observed.partition += volume * w;
            observed.density += source.mass * w;
            for (int axis = 0; axis < 3; ++axis)
                observed.gradient[axis] += volume * dz * gradient *
                                           (double(source.position[axis]) - position[axis]);
            ++observed.count;
        });
        const auto& expected = reference[(id - 1) / (fixture.n * fixture.n)];
        EXPECT_EQ(observed.count, expected.count);
        EXPECT_NEAR(observed.partition, expected.partition, 2e-13);
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_NEAR(observed.gradient[axis], expected.gradient[axis], 2e-13);
        EXPECT_NEAR(density, expected.density, field_tolerance * expected.density + 5e-10);
        EXPECT_NEAR(density, observed.density, field_tolerance * observed.density);
        const double expected_pressure =
            tait_b * std::expm1(eos.gamma * std::log(density / eos.reference_density));
        EXPECT_NEAR(pressure, expected_pressure, field_tolerance * tait_b + 1e-8);
        for (double component : ap)
            EXPECT_TRUE(std::isfinite(component));

        const double pressure_error =
            pressure - tait_b * std::expm1(eos.gamma * log_density_ratio(position[2]));
        const double acceleration_error = std::hypot(ap[0], ap[1], double(ap[2]) - gravity);
        total_mass += center.mass;
        pressure_error_sum += center.mass * pressure_error * pressure_error;
        acceleration_error_sum += center.mass * acceleration_error * acceleration_error;
        if (csv)
            *csv << fixture.name << ',' << fixture.n << ',' << dx << ',' << fixture.h << ',' << id
                 << ',' << position[0] << ',' << position[1] << ',' << position[2]
                 << ',' << observed.partition << ',' << observed.gradient[0]
                 << ',' << observed.gradient[1] << ',' << observed.gradient[2] << ',' << density
                 << ',' << pressure << ',' << ap[0] << ',' << ap[1] << ',' << ap[2]
                 << ',' << observed.count << '\n';
    }
    EXPECT_NEAR(std::sqrt(pressure_error_sum / total_mass), fixture.pressure_rms_pa, 2);
    EXPECT_NEAR(std::sqrt(acceleration_error_sum / total_mass) / gravity,
                fixture.acceleration_residual_rms_over_g, .002);
    ASSERT_TRUE(simulation.finalize().ok());
}

using LatticeIndex = std::array<int, 3>;
using WallSet = std::vector<LatticeIndex>;

WallSet extended_solid(int n) {
    // Extend beyond every test radius and clamp only at the open surface.
    // This source does not take the production shell's layer count as input.
    WallSet solid;
    constexpr int extension = 6;
    for (int x = -extension; x < n + extension; ++x)
        for (int y = -extension; y < n + extension; ++y)
            for (int z = -extension; z < n; ++z)
                if (x < 0 || x >= n || y < 0 || y >= n || z < 0)
                    solid.push_back({x, y, z});
    return solid;
}

WallSet expected_wall(const WallSet& solid, const std::array<real, 3>& position, double dx,
                      double h) {
    WallSet expected;
    const long double r2 = 4 * static_cast<long double>(h) * h;
    const auto x_center = [dx](const LatticeIndex& index) { return (index[0] + .5L) * dx; };
    const long double min_x = position[0] - 2 * static_cast<long double>(h);
    const long double max_x = position[0] + 2 * static_cast<long double>(h);
    auto candidate = std::lower_bound(solid.begin(), solid.end(), min_x,
                                      [&](const auto& index, auto x) {
                                          return x_center(index) < x;
                                      });
    for (; candidate != solid.end() && x_center(*candidate) < max_x; ++candidate) {
        const auto& index = *candidate;
        const long double x = x_center(index) - position[0];
        const long double y = (index[1] + .5L) * dx - position[1];
        const long double z = (index[2] + .5L) * dx - position[2];
        if (x * x + y * y + z * z < r2)
            expected.push_back(index);
    }
    return expected;
}

WallSet observed_wall(const sph::CellGrid& grid, std::size_t row, double dx) {
    WallSet observed;
    const auto& fields = grid.fields();
    grid.neighbors().for_each_neighbor(row, [&](LocalIndex slot, double d2) {
        if (fields.ghost(slot) && d2 < grid.support_radius() * grid.support_radius()) {
            LatticeIndex index;
            for (int axis = 0; axis < 3; ++axis)
                index[axis] = static_cast<int>(
                    std::lround(double(fields.position(slot, axis)) / dx - .5));
            observed.push_back(index);
        }
    });
    std::sort(observed.begin(), observed.end());
    return observed;
}

void compare_wall_sets(const sph::CellGrid& grid, const WallSet& solid, double dx, double h) {
    const auto& fields = grid.fields();
    for (std::size_t row = 0; row < grid.owned_slots().extent(0); ++row) {
        const auto slot = grid.owned_slots()(row);
        SCOPED_TRACE(fields.ids(slot));
        const std::array position{fields.position(slot, 0), fields.position(slot, 1),
                                  fields.position(slot, 2)};
        EXPECT_EQ(observed_wall(grid, row, dx), expected_wall(solid, position, dx, h));
    }
}

void check_shell(int n, double h, int layers) {
    SCOPED_TRACE(n);
    SCOPED_TRACE(h);
    const double dx = tank_size / n;
    EXPECT_EQ(wall_layers(h, dx), layers);
    const auto tank = make_tank(n, h);
    std::vector<Particle> all = tank.fluid;
    for (auto wall : tank.wall) {
        ASSERT_GT(wall.mass, 0);
        ASSERT_LT(wall.position[2], tank_size);
        for (const real coordinate : wall.position) {
            const auto index = std::lround(double(coordinate) / dx - .5);
            ASSERT_EQ(coordinate, static_cast<real>((index + .5) * dx));
        }
        wall.ghost = 1;
        all.push_back(wall);
    }
    const auto solid = extended_solid(n);
    const Kokkos::Serial exec;
    const auto grid = checked(sph::CellGrid::build(exec, Fields(all).get(), 10000, {}, h));
    compare_wall_sets(grid, solid, dx, h);

    // Move previously unsupported rows into side, side-edge and bottom-corner support.
    const real mid = 3, inside = static_cast<real>(dx / 16);
    const real aligned = static_cast<real>(mid - dx / 2);
    const std::array<std::array<real, 3>, 3> moved{
        std::array{inside, aligned, aligned}, std::array{inside, inside, mid},
        std::array{inside, inside, inside}};
    const auto first_probe = tank.fluid.size() / 2;
    for (std::size_t i = 0; i < moved.size(); ++i)
        all[first_probe + i].position = {mid, mid, mid};
    const Fields distant_fields(all);
    const auto distant = checked(grid.update(exec, distant_fields.get(), 10000));
    for (std::size_t i = 0; i < moved.size(); ++i)
        all[first_probe + i].position = moved[i];
    const auto near = checked(distant.update(exec, Fields(all).get(), 10000));
    const auto restored = checked(near.update(exec, distant_fields.get(), 10000));
    for (std::size_t row = 0; row < near.owned_slots().extent(0); ++row) {
        const auto id = near.fields().ids(near.owned_slots()(row));
        if (id <= first_probe || id > first_probe + moved.size())
            continue;
        const auto expected = expected_wall(solid, moved[id - first_probe - 1], dx, h);
        EXPECT_FALSE(expected.empty());
        EXPECT_EQ(observed_wall(near, row, dx), expected);
        if (layers == 3 && id == first_probe + 1)
            EXPECT_TRUE(std::any_of(expected.begin(), expected.end(),
                                   [](auto index) { return index[0] == -3; }));
    }
    for (const auto* outside : {&distant, &restored})
        for (std::size_t row = 0; row < outside->owned_slots().extent(0); ++row) {
            const auto id = outside->fields().ids(outside->owned_slots()(row));
            if (id > first_probe && id <= first_probe + moved.size())
                EXPECT_TRUE(observed_wall(*outside, row, dx).empty());
        }
}

TEST(SphHydrostaticInitial, FixedWallShellCoversNonzeroSupport) {
    for (const auto fixture : cases)
        ASSERT_NO_FATAL_FAILURE(check_shell(fixture.n, fixture.h, fixture.layers));
    ASSERT_NO_FATAL_FAILURE(check_shell(6, 1.3, 3));
}

TEST(SphHydrostaticInitial, AirSideStaysOpen) {
    constexpr int n = 6;
    constexpr double dx = tank_size / n, h = 1.3;
    const auto tank = make_tank(n, h);
    std::vector<Particle> all = tank.fluid;
    for (auto wall : tank.wall) {
        wall.ghost = 1;
        all.push_back(wall);
    }
    // At this rim point an extra top layer would contribute to the kernel.
    const std::array<real, 3> probe{.5, 2.5, 5.5};
    EXPECT_FALSE(expected_wall({LatticeIndex{0, 2, n}}, probe, dx, h).empty());
    const auto solid = extended_solid(n);
    const auto grid = checked(sph::CellGrid::build(Kokkos::Serial{}, Fields(all).get(), 10000, {}, h));
    for (std::size_t row = 0; row < grid.owned_slots().extent(0); ++row) {
        const auto slot = grid.owned_slots()(row);
        const auto& p = grid.fields().position;
        if (std::array{p(slot, 0), p(slot, 1), p(slot, 2)} == probe) {
            const auto observed = observed_wall(grid, row, dx);
            ASSERT_FALSE(observed.empty());
            EXPECT_EQ(observed, expected_wall(solid, probe, dx, h));
            EXPECT_TRUE(std::all_of(observed.begin(), observed.end(),
                                    [](auto index) { return index[2] < n; }));
            return;
        }
    }
    FAIL() << "missing rim probe";
}

TEST(SphHydrostaticInitial, ConventionalMassFourCases) {
    std::ofstream csv;
    const char* path = std::getenv("NEXORA_SPH_HYDROSTATIC_INITIAL_CSV");
    if (path) {
        csv.open(path);
        ASSERT_TRUE(csv.is_open()) << path;
        csv.imbue(std::locale::classic());
        csv << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "case,n,dx,h,id,x,y,z,S,grad_x,grad_y,grad_z,rho,p,ap_x,ap_y,ap_z,neighbor_count\n";
    }
    for (const auto fixture : cases)
        ASSERT_NO_FATAL_FAILURE(measure(fixture, path ? &csv : nullptr));
    if (path) {
        csv.close();
        ASSERT_TRUE(csv.good()) << path;
    }
}
} // namespace
