#include "cell_grid_oracle.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <nexora/simulation/Simulation.hpp>
#include <numbers>
#include <participants/sph/Participant3D.hpp>

namespace {
using namespace nexora;
using namespace sph_test;

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

// The local all-pairs reference keeps the wall at its original coordinates in
// every snapshot. A detached dry wall exercises the empty support case.
void check_wall_forces(const sph::Snapshot3D& snapshot, std::vector<Particle> particles,
                       const sph::WcsphVerletParameters& parameters) {
    constexpr long double pi = std::numbers::pi_v<long double>;
    std::map<std::uint64_t, long double> rho, pressure;
    std::map<std::uint64_t, std::size_t> rows;
    for (std::size_t row = 0; row < snapshot.row_count(); ++row)
        ASSERT_TRUE(rows.emplace(snapshot.row_id(row), row).second);
    for (auto& p : particles)
        if (!p.ghost) {
            ASSERT_TRUE(rows.contains(p.id));
            p.position = checked(snapshot.position(rows.at(p.id)));
        }
    const long double rho0 = parameters.eos.reference_density;
    const long double gamma = parameters.eos.gamma;
    const long double bulk = rho0 * std::pow(parameters.eos.sound_speed, 2) / gamma;
    // Independently re-evaluate the EOS, wall extrapolation and pair forces at
    // the evolved fluid density. Continuity has its own time-integration oracle.
    for (const auto& [id, row] : rows) {
        rho[id] = snapshot.density(row);
        pressure[id] = static_cast<real>(bulk * (std::pow(rho[id] / rho0, gamma) - 1));
    }
    const auto distance = [](const Particle& a, const Particle& b) {
        return std::hypot(static_cast<long double>(a.position[0]) - b.position[0],
                          static_cast<long double>(a.position[1]) - b.position[1],
                          static_cast<long double>(a.position[2]) - b.position[2]);
    };
    for (const auto& wall : particles) {
        if (!wall.ghost)
            continue;
        long double weight_sum{}, weighted_pressure{};
        for (const auto& f : particles) {
            const auto r = distance(wall, f);
            if (f.ghost || r >= 2)
                continue;
            const auto weight =
                (std::pow(2 - r, 3) - 4 * std::pow(std::max(1 - r, 0.L), 3)) / (4 * pi);
            long double head{};
            for (int k = 0; k < 3; ++k)
                head += parameters.gravity[k] *
                        (static_cast<long double>(wall.position[k]) - f.position[k]);
            weight_sum += weight;
            weighted_pressure += weight * (pressure.at(f.id) + rho.at(f.id) * head);
        }
        if (weight_sum > 0) {
            pressure[wall.id] = weighted_pressure / weight_sum;
            rho[wall.id] = rho0 * std::pow(1 + pressure.at(wall.id) / bulk, 1 / gamma);
        }
    }
    for (const auto& f : particles) {
        if (f.ghost)
            continue;
        std::array<long double, 3> a{};
        for (const auto& other : particles) {
            const auto r = distance(f, other);
            if (r == 0 || r >= 2)
                continue;
            const auto grad = r < 1 ? (3 - 2.25L * r) / pi : .75L * (2 - r) * (2 - r) / (pi * r);
            const auto coefficient = other.mass * grad *
                                     (pressure.at(f.id) / std::pow(rho.at(f.id), 2) +
                                      pressure.at(other.id) / std::pow(rho.at(other.id), 2));
            for (int k = 0; k < 3; ++k)
                a[k] += coefficient * (static_cast<long double>(f.position[k]) - other.position[k]);
        }
        const auto row = rows.at(f.id);
        const auto epsilon = std::numeric_limits<real>::epsilon();
        EXPECT_NEAR(checked(snapshot.pressure(row)), pressure.at(f.id), 64 * epsilon * bulk);
        const auto observed = checked(snapshot.pressure_acceleration(row));
        for (int k = 0; k < 3; ++k)
            EXPECT_NEAR(observed[k], a[k], 128 * epsilon * std::max(bulk / rho0, std::abs(a[k])));
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
    fixed.position(0, 2) = 10; // caller may reuse these buffers after initialization
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
