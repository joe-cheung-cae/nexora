#include "cell_grid_oracle.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <methods/sph/AdamiWallPressure3D.hpp>
#include <nexora/simulation/Simulation.hpp>
#include <numbers>
#include <participants/sph/Participant3D.hpp>

namespace {
using namespace nexora;
using namespace sph_test;

TEST(SphAdamiWall, InitialAdamiFieldsAndAcousticBoundScaleWithSmoothingLength) {
    struct Initial {
        double density, pressure, acceleration;
    };
    std::array<Initial, 2> reference{};
    for (const real h : {real(1), real(.5), real(2)}) {
        const real volume = h * h * h;
        Fields fluid({{{0, 0, 0}, volume, 10, 0}, {{real(.75) * h, 0, 0}, 2 * volume, 20, 0}});
        Fields wall({{{real(-.5) * h, 0, 0}, volume, 30, 1}});
        const sph::FixedWallParticles3D boundary{wall.position, wall.mass, wall.ids};
        sph::Velocity3D velocity("test.h_scaled_velocity", 2);
        velocity(0, 0) = 1;
        const sph::WcsphVerletParameters parameters{{.5, 3, 1}, .25, {-.125 / h, 0, 0}, h};
        sph::Values density("test.scaled_physical_density", 2);
        density(0) = fluid.mass(0) / volume;
        density(1) = fluid.mass(1) / volume;
        sph::Participant3D participant(fluid.get(), 100, parameters, velocity, density, boundary);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        const auto snapshot = checked(participant.snapshot());
        ASSERT_EQ(snapshot.row_count(), reference.size());
        EXPECT_EQ(participant.time_constraint(0).maximum_step, .25 * h / 4);
        for (std::size_t row = 0; row < reference.size(); ++row) {
            const auto i = snapshot.row_id(row) == 10 ? 0 : 1;
            const Initial observed{snapshot.density(row), checked(snapshot.pressure(row)),
                                   h * checked(snapshot.pressure_acceleration(row))[0]};
            if (h == 1) {
                reference[i] = observed;
                EXPECT_NE(observed.acceleration, 0);
            } else {
                EXPECT_EQ(observed.density, reference[i].density);
                EXPECT_EQ(observed.pressure, reference[i].pressure);
                EXPECT_EQ(observed.acceleration, reference[i].acceleration);
            }
        }
        ASSERT_TRUE(simulation.finalize().ok());
    }
}

TEST(SphAdamiWall, GravityOnlyMatchesFreeFall) {
    Fields fluid({{{0, 0, 0}, 1, 41, 0}});
    sph::Velocity3D velocity("wall_test.initial_velocity", 1);
    const std::array<double, 3> gravity{1, -2, -4};
    const std::array<double, 3> initial_velocity{2, 1, -1};
    for (int axis = 0; axis < 3; ++axis)
        velocity(0, axis) = static_cast<real>(initial_velocity[axis]);
    sph::Participant3D participant(fluid.get(), 512, {{1, 2, 2}, .8, gravity}, velocity, fluid.mass);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    for (int step = 1; step <= 16; ++step) {
        constexpr double dt = 1. / 32;
        ASSERT_TRUE(simulation.advance(dt).ok());
        const double t = step * dt;
        const auto snapshot = checked(participant.snapshot());
        const auto x = checked(snapshot.position(0));
        const auto v = checked(snapshot.velocity(0));
        EXPECT_EQ(checked(snapshot.pressure_acceleration(0)), (std::array<real, 3>{}));
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_EQ(v[axis], initial_velocity[axis] + gravity[axis] * t);
            EXPECT_EQ(x[axis], initial_velocity[axis] * t + .5 * gravity[axis] * t * t);
        }
    }
    EXPECT_TRUE(simulation.finalize().ok());
}

struct WallForceReference {
    std::map<std::uint64_t, real> fluid_pressure;
    std::map<std::uint64_t, long double> wall_pressure;
    std::map<std::uint64_t, double> coefficient;
    std::map<std::uint64_t, std::array<long double, 3>> acceleration;
};

WallForceReference independent_wall_forces(
    const std::vector<Particle>& particles, const std::map<std::uint64_t, real>& density,
    const sph::WcsphVerletParameters& parameters) {
    constexpr long double pi = std::numbers::pi_v<long double>;
    const long double rho0 = parameters.eos.reference_density;
    const long double gamma = parameters.eos.gamma;
    const long double h = parameters.smoothing_length;
    const long double bulk = rho0 * std::pow(parameters.eos.sound_speed, 2) / gamma;
    const auto distance = [](const Particle& a, const Particle& b) {
        return std::hypot(static_cast<long double>(a.position[0]) - b.position[0],
                          static_cast<long double>(a.position[1]) - b.position[1],
                          static_cast<long double>(a.position[2]) - b.position[2]);
    };
    const auto coefficient = [](real p, real rho) { return double(p) / (double(rho) * rho); };
    WallForceReference result;
    for (const auto& [id, rho] : density) {
        const auto p = static_cast<real>(bulk * (std::pow(rho / rho0, gamma) - 1));
        result.fluid_pressure.emplace(id, p);
        result.coefficient.emplace(id, coefficient(p, rho));
    }
    for (const auto& wall : particles) {
        if (!wall.ghost)
            continue;
        long double weight_sum{}, weighted_pressure{};
        for (const auto& fluid : particles) {
            const auto q = distance(wall, fluid) / h;
            if (fluid.ghost || q >= 2)
                continue;
            const auto weight =
                (std::pow(2 - q, 3) - 4 * std::pow(std::max(1 - q, 0.L), 3)) / (4 * pi);
            long double head{};
            for (int k = 0; k < 3; ++k)
                head += parameters.gravity[k] *
                        (static_cast<long double>(wall.position[k]) - fluid.position[k]);
            weight_sum += weight;
            weighted_pressure += weight *
                (result.fluid_pressure.at(fluid.id) + density.at(fluid.id) * head);
        }
        if (weight_sum > 0) {
            const long double raw = weighted_pressure / weight_sum;
            const auto p = static_cast<real>(raw);
            const auto rho = static_cast<real>(rho0 * std::pow(1 + p / bulk, 1 / gamma));
            result.wall_pressure.emplace(wall.id, raw);
            result.coefficient.emplace(wall.id, coefficient(p, rho));
        }
    }
    for (const auto& fluid : particles) {
        if (fluid.ghost)
            continue;
        std::array<long double, 3> acceleration{};
        for (const auto& other : particles) {
            const auto q = distance(fluid, other) / h;
            if (q == 0 || q >= 2)
                continue;
            const auto grad = q < 1 ? (3 - 2.25L * q) / pi : .75L * (2 - q) * (2 - q) / (pi * q);
            const auto pair = other.mass * grad / std::pow(h, 5) *
                              (result.coefficient.at(fluid.id) + result.coefficient.at(other.id));
            for (int k = 0; k < 3; ++k)
                acceleration[k] += pair *
                    (static_cast<long double>(fluid.position[k]) - other.position[k]);
        }
        result.acceleration.emplace(fluid.id, acceleration);
    }
    return result;
}

void expect_wall_reference(const sph::Snapshot3D& snapshot, const WallForceReference& reference,
                           const sph::WcsphVerletParameters& parameters) {
    const double eps = std::numeric_limits<real>::epsilon();
    const double c2 = parameters.eos.sound_speed * parameters.eos.sound_speed;
    const double pressure_scale = parameters.eos.reference_density * c2 / parameters.eos.gamma;
    ASSERT_EQ(snapshot.row_count(), reference.fluid_pressure.size());
    for (std::size_t row = 0; row < snapshot.row_count(); ++row) {
        const auto id = snapshot.row_id(row);
        ASSERT_TRUE(reference.fluid_pressure.contains(id));
        EXPECT_NEAR(checked(snapshot.pressure(row)), reference.fluid_pressure.at(id),
                    64 * eps * pressure_scale);
        const auto observed = checked(snapshot.pressure_acceleration(row));
        const auto& expected = reference.acceleration.at(id);
        for (int k = 0; k < 3; ++k)
            EXPECT_NEAR(observed[k], expected[k], 128 * eps *
                        std::max<long double>(c2 / parameters.eos.gamma, std::abs(expected[k])));
    }
}

// The all-pairs reference leaves the detached dry wall at its original coordinates.
void check_wall_forces(const sph::Snapshot3D& snapshot, std::vector<Particle> particles,
                       const sph::WcsphVerletParameters& parameters) {
    std::map<std::uint64_t, real> density;
    std::map<std::uint64_t, std::size_t> rows;
    for (std::size_t row = 0; row < snapshot.row_count(); ++row) {
        ASSERT_TRUE(rows.emplace(snapshot.row_id(row), row).second);
        density.emplace(snapshot.row_id(row), snapshot.density(row));
    }
    for (auto& particle : particles)
        if (!particle.ghost) {
            ASSERT_TRUE(rows.contains(particle.id));
            particle.position = checked(snapshot.position(rows.at(particle.id)));
        }
    expect_wall_reference(snapshot, independent_wall_forces(particles, density, parameters),
                          parameters);
}

TEST(SphAdamiWall, WallPressureAndForcesMatchIndependentReference) {
    const std::vector<Particle> fluid{
        {{.875, .5, 0}, 2.5, 20, 0}, {{.125, 0, 0}, 1.5, 40, 0}};
    const Particle wall{{.125, 0, 0}, 4, 99, 1};
    auto particles = fluid;
    particles.push_back(wall);
    const sph::WcsphVerletParameters parameters{{1, 2, 3}, .75, {-.25, .125, -.375}, 1.25};
    const sph::TaitEos eos(parameters.eos);
    const std::array<double, 3> fixed_acceleration{};
    const Kokkos::Serial exec;
    const auto grid = checked(sph::CellGrid::build(exec, Fields(particles).get(), 64, {},
                                                   parameters.smoothing_length));
    const auto& sources = grid.fields();
    const auto owned = grid.owned_slots();
    sph::CellGrid::Indices::non_const_type row_of_slot("wall_test.row_of_slot", particles.size());
    LocalIndex wall_slot{};
    for (std::size_t slot = 0; slot < particles.size(); ++slot)
        if (sources.ids(slot) == wall.id)
            wall_slot = static_cast<LocalIndex>(slot);
    ASSERT_EQ(sources.ids(wall_slot), wall.id);
    for (std::size_t row = 0; row < fluid.size(); ++row)
        row_of_slot(owned(row)) = static_cast<LocalIndex>(row);

    const Fields fluid_fields(fluid), wall_fields({wall});
    const sph::FixedWallParticles3D boundary{wall_fields.position, wall_fields.mass, wall_fields.ids};
    const sph::Velocity3D velocity("wall_test.velocity", fluid.size());
    for (const real coincident_density : {real{1.125}, real{1.375}}) {
        SCOPED_TRACE(coincident_density);
        // The offset fluid marks the wall. Only the coincident fluid's density changes.
        const std::map<std::uint64_t, real> density{{20, real{1.25}}, {40, coincident_density}};
        const auto reference = independent_wall_forces(particles, density, parameters);
        sph::Values owned_density("wall_test.owned_density", fluid.size());
        sph::Values owned_pressure("wall_test.owned_pressure", fluid.size());
        sph::Values input_density("wall_test.input_density", fluid.size());
        for (std::size_t row = 0; row < fluid.size(); ++row) {
            const auto id = sources.ids(owned(row));
            owned_density(row) = density.at(id);
            owned_pressure(row) = reference.fluid_pressure.at(id);
            input_density(row) = density.at(fluid[row].id);
        }
        const double raw = sph::wall_pressure(grid, wall_slot, row_of_slot, owned_density,
                                               owned_pressure, fixed_acceleration, parameters.gravity);
        const auto coefficient = sph::wall_coefficient(grid, wall_slot, row_of_slot, owned_density,
                                                       owned_pressure, fixed_acceleration,
                                                       parameters.gravity, eos);
        ASSERT_TRUE(coefficient.has_value());
        EXPECT_NEAR(raw, reference.wall_pressure.at(wall.id),
                    32 * std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(*coefficient, reference.coefficient.at(wall.id),
                    16 * std::numeric_limits<real>::epsilon());

        sph::Participant3D participant(fluid_fields.get(), 64, parameters, velocity, input_density,
                                       boundary);
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        expect_wall_reference(checked(participant.snapshot()), reference, parameters);
        EXPECT_TRUE(simulation.finalize().ok());
    }
}

TEST(SphAdamiWall, FixedSourcesMatchIndependentWallForcesAcrossRestore) {
    std::vector<Particle> fluid{{{1.875, .25, .5}, 2, 33, 0}, {{1, .5, .5}, 3, 11, 0}};
    const std::vector<Particle> walls{
        {{1.5, 0, -.25}, 2, 44, 1}, {{.5, -.5, .25}, 1, 20, 1}, {{12, 0, 0}, 2, 55, 1}};
    Fields input(fluid), fixed(walls);
    sph::Velocity3D velocity("wall_test.velocity", fluid.size());
    velocity(0, 0) = 5;
    const sph::WcsphVerletParameters parameters{{1, 2, 3}, .75, {0, -.25, -1}};
    const sph::FixedWallParticles3D boundary{fixed.position, fixed.mass, fixed.ids};
    sph::Values initial_density("wall_test.initial_density", fluid.size());
    initial_density(0) = real{1.125};
    initial_density(1) = real{1.25};
    sph::Participant3D participant(input.get(), 4096, parameters, velocity, initial_density, boundary);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    // Caller may reuse all wall input buffers after initialization.
    fixed.position(0, 2) = 10;
    fixed.mass(0) = 7;
    fixed.ids(0) = 33;
    fluid.insert(fluid.end(), walls.begin(), walls.end());
    const auto initial = checked(participant.snapshot());
    ASSERT_EQ(initial.row_count(), 2u);
    ASSERT_NO_FATAL_FAILURE(check_wall_forces(initial, fluid, parameters));
    for (std::size_t row = 0; row < initial.row_count(); ++row) {
        const double vx = initial.row_id(row) == 33 ? 5 : 0;
        EXPECT_EQ(initial.density(row), initial.row_id(row) == 33 ? 1.125 : 1.25);
        EXPECT_EQ(checked(initial.velocity(row))[0], vx);
    }
    ASSERT_TRUE(simulation.begin_window(.05).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto trial = checked(participant.trial_snapshot(checked(participant.trial_stamp())));
    ASSERT_NO_FATAL_FAILURE(check_wall_forces(trial, fluid, parameters));
    ASSERT_TRUE(simulation.restore_window().ok());
    const auto restored = checked(participant.snapshot());
    for (std::size_t row = 0; row < initial.row_count(); ++row) {
        EXPECT_EQ(restored.row_id(row), initial.row_id(row));
        EXPECT_EQ(restored.density(row), initial.density(row));
        EXPECT_EQ(checked(restored.position(row)), checked(initial.position(row)));
        EXPECT_EQ(checked(restored.velocity(row)), checked(initial.velocity(row)));
    }
    ASSERT_TRUE(simulation.advance(.025).ok());
    ASSERT_NO_FATAL_FAILURE(check_wall_forces(checked(participant.snapshot()), fluid, parameters));
    ASSERT_NO_FATAL_FAILURE(check_wall_forces(trial, fluid, parameters));
    EXPECT_TRUE(simulation.finalize().ok());
}

TEST(SphAdamiWall, RejectsInvalidGravityWallIdentityAndGhostFluid) {
    Fields input({{{0, 0, 0}, 1, 12, 0}}), wall({{{0, .25, 1}, 2, 21, 0}});
    sph::Velocity3D velocity("wall_test.validation_velocity", 1);
    const sph::FixedWallParticles3D boundary{wall.position, wall.mass, wall.ids};
    sph::WcsphVerletParameters parameters{{1, 2, 3}, .5};
    parameters.gravity[1] = std::numeric_limits<double>::infinity();
    sph::Participant3D invalid_g(input.get(), 128, parameters, velocity, input.mass, boundary);
    EXPECT_FALSE(invalid_g.initialize(0).ok());
    parameters.gravity[1] = 0;
    wall.ids(0) = 12;
    sph::Participant3D duplicate(input.get(), 128, parameters, velocity, input.mass, boundary);
    EXPECT_FALSE(duplicate.initialize(0).ok());
    wall.ids(0) = 21;
    input.ghost(0) = 1;
    sph::Participant3D ghost(input.get(), 128, parameters, velocity, input.mass, boundary);
    EXPECT_FALSE(ghost.initialize(0).ok());
}

TEST(SphAdamiWall, InverseEosFailureRetainsAcceptedStateForShorterRetry) {
    Fields fluid({{{0, 0, 0}, 1, 1, 0}}), wall({{{0, 0, 2.125}, 1, 2, 0}});
    sph::Velocity3D velocity("wall_test.approach", 1);
    velocity(0, 2) = 3;
    const sph::WcsphVerletParameters parameters{{1, 1, 1}, 1, {0, 0, -10}};
    const sph::FixedWallParticles3D boundary{wall.position, wall.mass, wall.ids};
    sph::Participant3D participant(fluid.get(), 128, parameters, velocity, fluid.mass, boundary);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto accepted = checked(participant.snapshot());
    // The longer step brings an overhead wall into support. Its extrapolated
    // pressure falls below -B, where the wall has no positive Tait density.
    const auto status = simulation.advance(.1);
    EXPECT_EQ(status.code(), StatusCode::numerical_failure);
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0);
    EXPECT_EQ(checked(checked(participant.snapshot()).position(0)), checked(accepted.position(0)));
    ASSERT_TRUE(simulation.advance(.01).ok());
    EXPECT_NEAR(checked(checked(participant.snapshot()).velocity(0))[2], 2.9,
                4 * std::numeric_limits<real>::epsilon());
    EXPECT_TRUE(simulation.finalize().ok());
}
} // namespace
