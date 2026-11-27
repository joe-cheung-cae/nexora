#include "cell_grid_oracle.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <methods/sph/Continuity3D.hpp>
#include <nexora/simulation/Simulation.hpp>
#include <numbers>
#include <participants/sph/WcsphParticipant3D.hpp>
#include <sstream>
#include <vector>

namespace {
using namespace nexora;
using namespace sph_test;

sph::Values initial_density(std::initializer_list<real> entries) {
    sph::Values result("continuity_test.density", entries.size());
    std::copy(entries.begin(), entries.end(), result.data());
    return result;
}

std::vector<sph::CommittedRow3D> committed_rows(sph::WcsphParticipant3D& participant) {
    std::vector<sph::CommittedRow3D> rows;
    const auto status = participant.visit_committed_rows(
        [&](sph::CommittedRow3D row) { rows.push_back(row); });
    EXPECT_TRUE(status.ok()) << status.message();
    return rows;
}

TEST(SphVerlet, SourceCapacityPrecedesOwnershipAndWallCapture) {
    const Fields fluid({{{0, 0, 0}, 1, 11, 1}});
    const sph::Velocity3D velocity("test.capacity.velocity", 1);
    const auto density = initial_density({1});
    const std::size_t capacity = std::numeric_limits<LocalIndex>::max();
    const std::uint64_t id = 21;
    sph::FixedWallParticles3D oversized;
    // Count admission uses only this extent, before wall storage or records are read.
    oversized.ids = decltype(oversized.ids)(&id, capacity);
    for (const bool empty_group : {false, true}) {
        std::vector<sph::SphBoundary::WallGroupInput3D> walls;
        if (empty_group)
            walls.push_back({0, {}, std::nullopt});
        walls.push_back({1, oversized, std::nullopt});
        sph::WcsphParticipant3D participant(fluid.get(), 1, {{1, 1, 1}, .5},
                                           velocity, density, std::move(walls));
        const auto failure = participant.initialize(0);
        EXPECT_EQ(failure.code(), StatusCode::resource_exhausted);
        EXPECT_EQ(failure.message(), "WCSPH source count exceeds capacity");
        EXPECT_TRUE(participant.finalize().ok());
    }
}

TEST(SphVerlet, FluidExtentValidationPrecedesSourceCapacity) {
    const Fields fluid({{{0, 0, 0}, 1, 11, 1}});
    const std::uint64_t id = 21;
    sph::FixedWallParticles3D oversized;
    oversized.ids = decltype(oversized.ids)(&id, std::numeric_limits<LocalIndex>::max());
    sph::WcsphParticipant3D participant(fluid.get(), 1, {{1, 1, 1}, .5}, {}, {},
                                       std::vector<sph::SphBoundary::WallGroupInput3D>{{1, oversized, std::nullopt}});
    const auto failure = participant.initialize(0);
    EXPECT_EQ(failure.code(), StatusCode::invalid_argument);
    EXPECT_EQ(failure.message(), "WCSPH fluid field extents disagree");
    EXPECT_TRUE(participant.finalize().ok());
}

TEST(SphVerletOperators, KickDriftWritesOwnedSlotsAndPreservesHalfStepFormula) {
    const Kokkos::Serial exec;
    const Fields fields({{{1, .5, .25}, 1, 11, 0}, {{1.5, 1, .75}, 2, 22, 1}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 1));
    sph::Velocity3D velocity("test.kick.velocity", 1), acceleration("test.kick.acceleration", 1);
    sph::Velocity3D position("test.kick.position", 2), half("test.kick.half", 2);
    velocity(0, 0) = 2;
    velocity(0, 1) = -1;
    velocity(0, 2) = .5;
    acceleration(0, 0) = 4;
    acceleration(0, 1) = 2;
    acceleration(0, 2) = -2;
    constexpr std::array<real, 3> expected_half{2.5, -.75, .25};
    constexpr std::array<real, 3> expected_position{1.625, .3125, .3125};
    Kokkos::deep_copy(exec, position, real{-19});
    Kokkos::deep_copy(exec, half, real{-17});
    const auto result = sph::kick_drift_3d(exec, grid, velocity, acceleration, .25, position, half);
    ASSERT_TRUE(result.ok()) << result.message();
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_EQ(half(0, axis), expected_half[axis]);
        EXPECT_EQ(position(0, axis), expected_position[axis]);
        EXPECT_EQ(half(1, axis), real{-17});
        EXPECT_EQ(position(1, axis), real{-19});
    }
    Kokkos::deep_copy(exec, position, real{-19});
    sph::central_difference_step::drift_positions(exec, grid, velocity, acceleration, .25, position, {});
    for (int axis = 0; axis < 3; ++axis) {
        EXPECT_EQ(position(0, axis), expected_position[axis]);
        EXPECT_EQ(position(1, axis), real{-19});
    }
}

TEST(SphVerletOperators, StandaloneKickDriftRejectsOverflowWithoutGridAdmission) {
    const Kokkos::Serial exec;
    const Fields fields({{{0, 0, 0}, 1, 11, 0}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 1));
    sph::Velocity3D velocity("test.kick.velocity", 1), acceleration("test.kick.acceleration", 1);
    sph::Velocity3D position("test.kick.position", 1), half("test.kick.half", 1);
    const real largest = std::numeric_limits<real>::max();
    // Finite inputs overflow the half kick, then separately the position drift.
    for (const bool overflow_half : {true, false}) {
        Kokkos::deep_copy(exec, velocity, real{0});
        Kokkos::deep_copy(exec, acceleration, real{0});
        velocity(0, 0) = overflow_half ? real{0} : largest / real{4};
        acceleration(0, 0) = overflow_half ? largest : real{0};
        const double dt = overflow_half ? 4 : 8;
        EXPECT_EQ(sph::kick_drift_3d(exec, grid, velocity, acceleration, dt, position, half).code(),
                  StatusCode::numerical_failure);
    }
}

TEST(SphVerletOperators, StepRejectsPositionOverflowAtGridBoundary) {
    const Kokkos::Serial exec;
    const Fields fields({{{0, 0, 0}, 1, 11, 0}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 1));
    sph::Velocity3D velocity("test.kick.velocity", 1), acceleration("test.kick.acceleration", 1);
    sph::Velocity3D position("test.kick.position", 1);
    for (const bool overflow_half : {true, false}) {
        Kokkos::deep_copy(exec, velocity, real{0});
        Kokkos::deep_copy(exec, acceleration, real{0});
        velocity(0, 0) = overflow_half ? 0 : std::numeric_limits<real>::max() / 4;
        acceleration(0, 0) = overflow_half ? std::numeric_limits<real>::max() : 0;
        const double dt = overflow_half ? 4 : 8;
        sph::central_difference_step::drift_positions(exec, grid, velocity, acceleration, dt, position, {});
        EXPECT_FALSE(std::isfinite(position(0, 0)));
        EXPECT_EQ(grid.update_positions(exec, position, 1).status().code(),
                  StatusCode::invalid_argument);
        EXPECT_EQ(grid.fields().position(0, 0), 0);
    }
}

TEST(SphVerlet, UniformAccelerationMatchesBallisticTrajectory) {
    constexpr std::array<real, 3> position{1, .5, .25}, initial_velocity{2, -1, .5};
    const Fields fields({{position, 1, 11, 0}});
    sph::Velocity3D velocity("test.velocity", 1);
    for (int axis = 0; axis < 3; ++axis)
        velocity(0, axis) = initial_velocity[axis];
    const sph::WcsphVerletParameters parameters{{1, 1, 1}, 1, {4, 2, -2}};
    sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, initial_density({1}));
    Simulation sim(participant);
    ASSERT_TRUE(sim.initialize().ok());
    double t = 0;
    for (int step = 0; step < 16; ++step) {
        SCOPED_TRACE(step);
        const double dt = step == 0 ? .25 : 1. / 32;
        ASSERT_TRUE(sim.run(dt, 1).ok());
        t += dt;
        const auto rows = committed_rows(participant);
        ASSERT_EQ(rows.size(), 1u);
        const auto& row = rows.front();
        EXPECT_EQ(row.id, 11u);
        EXPECT_EQ(row.pressure, 0);
        EXPECT_EQ(row.pressure_acceleration, (std::array<real, 3>{}));
        for (int axis = 0; axis < 3; ++axis) {
            const double g = parameters.gravity[axis];
            EXPECT_EQ(row.position[axis], position[axis] + initial_velocity[axis] * t + .5 * g * t * t);
            EXPECT_EQ(row.velocity[axis], initial_velocity[axis] + g * t);
        }
    }
    EXPECT_TRUE(sim.finalize().ok());
}

TEST(SphVerlet, PositionOverflowPreservesRowsAndAllowsShorterRetry) {
    const Fields fields({{{0, 0, 0}, 1, 11, 0}});
    const sph::Velocity3D velocity("test.velocity", 1);
    constexpr real gravity = std::numeric_limits<real>::max();
    const sph::WcsphVerletParameters parameters{{1, 1, 1}, 1, {gravity, 0, 0}, 8};
    sph::WcsphParticipant3D participant(fields.get(), 1, parameters, velocity, initial_density({1}));
    ASSERT_TRUE(participant.initialize(0).ok());
    const auto before = committed_rows(participant);
    ASSERT_EQ(before.size(), 1u);

    const auto bound = participant.time_constraint().maximum_step;
    ASSERT_GE(bound, 4);
    const auto failure = participant.step(TimeInterval{0, 4, 4});
    EXPECT_EQ(failure.code(), StatusCode::numerical_failure);
    EXPECT_EQ(failure.message(), "invalid drifted WCSPH position");
    EXPECT_EQ(participant.time_constraint().maximum_step, bound);
    EXPECT_EQ(committed_rows(participant), before);

    const double dt = std::ldexp(1.0, 2 - std::numeric_limits<real>::max_exponent);
    ASSERT_TRUE(participant.step(TimeInterval{0, dt, dt}).ok());
    const auto retried = committed_rows(participant);
    ASSERT_EQ(retried.size(), 1u);
    EXPECT_EQ(retried[0].velocity[0], dt * gravity);
    EXPECT_TRUE(participant.finalize().ok());
}

TEST(SphVerlet, NonfiniteVelocityAmongFiniteRowsRejectsInitialization) {
    const Fields fields({{{0, 0, 0}, 1, 11, 0}, {{.5, 0, 0}, 1, 22, 0}});
    sph::Velocity3D velocity("test.velocity", 2);
    velocity(0, 1) = std::numeric_limits<real>::quiet_NaN();
    velocity(1, 2) = 1;
    sph::WcsphParticipant3D participant(fields.get(), 2, {{1, 1, 1}, 1}, velocity,
                                        initial_density({1, 1}));
    Simulation sim(participant);
    const auto failure = sim.initialize();
    EXPECT_EQ(failure.code(), StatusCode::invalid_argument);
    EXPECT_EQ(failure.message(), "WCSPH initial squared speed must be finite");
    EXPECT_EQ(participant.visit_committed_rows([](sph::CommittedRow3D) {}).code(),
              StatusCode::invalid_state);
    EXPECT_TRUE(sim.finalize().ok());
}

TEST(SphContinuity, NonfiniteDensityRateRejectsCandidate) {
    Kokkos::Serial exec;
    Fields fields({{{0, 0, 0}, std::numeric_limits<real>::max(), 31, 0}, {{1, 0, 0}, 1, 42, 0}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 2));
    sph::DensityRate3D rate("test.density_rate", 2);
    sph::Velocity3D velocity("test.velocity", 2);
    velocity(0, 0) = 8;
    const auto status = sph::evaluate_density_rate_3d(exec, grid.neighbors(), velocity, {}, rate);
    EXPECT_EQ(status.code(), StatusCode::numerical_failure);
    EXPECT_EQ(status.message(), "WCSPH density rate exceeds real range");
}

TEST(SphContinuity, CurrentRateSamplesOnlyContributingWalls) {
    Kokkos::Serial exec;
    const real tiny = std::numeric_limits<real>::denorm_min();
    Fields fields({{{0, 0, 0}, 1, 10, 0}, {{0, 0, 0}, 1, 11, 1}, {{2, 0, 0}, 1, 12, 1},
                   {{10, 0, 0}, 1, 20, 0}, {{10.5, 0, 0}, 1, 21, 0}, {{11, 0, 0}, 1, 22, 1},
                   {{20, 0, 0}, 1, 30, 0}, {{0, 10, 0}, 1, 40, 0}, {{tiny, 10, 0}, 1, 41, 1}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 128));
    const auto& query = grid.neighbors();
    const auto& ids = query.fields().ids;
    const auto& owned = query.owned_slots();
    sph::Velocity3D wall_velocity("test.wall_velocity", ids.extent(0));
    Kokkos::deep_copy(exec, wall_velocity, real{77});
    sph::Velocity3D velocity("test.owned_velocity", owned.extent(0));
    sph::NeighborView3D::Indices::non_const_type slot_to_row("test.slot_to_row", ids.extent(0));
    for (std::size_t row = 0; row < owned.extent(0); ++row) {
        const auto slot = owned(row);
        slot_to_row(slot) = static_cast<LocalIndex>(row);
        velocity(row, 0) = ids(slot) == 20 ? real{1} : real{0};
    }
    sph::StageVelocityReady3D ready("test.wall_velocity_ready", ids.extent(0));
    sph::DensityRate3D rate("test.density_rate", owned.extent(0));
    std::vector<std::uint64_t> read_walls;
    const auto write = [&](LocalIndex slot) {
        const auto id = ids(slot);
        read_walls.push_back(id);
        if (id != 22)
            return false;
        wall_velocity(slot, 0) = real{-.5};
        wall_velocity(slot, 1) = wall_velocity(slot, 2) = real{0};
        return true;
    };
    ASSERT_TRUE(sph::evaluate_density_rate_3d(exec, query, velocity, slot_to_row, rate,
                                             wall_velocity, ready, std::cref(write)).ok());
    // The displacement of wall 41 squares to zero. Both contributing fluid rows reuse wall 22.
    EXPECT_EQ(read_walls, (std::vector<std::uint64_t>{22}));
    for (std::size_t slot = 0; slot < ids.extent(0); ++slot)
        EXPECT_EQ(wall_velocity(slot, 0), ids(slot) == 22 ? real{-.5} : real{77});
    for (std::size_t row = 0; row < rate.extent(0); ++row) {
        const auto id = ids(owned(row));
        const double expected = (id == 20 ? 2.0625 : id == 21 ? 1.40625 : 0.) / std::numbers::pi;
        EXPECT_NEAR(rate(row), expected, 4 * std::numeric_limits<real>::epsilon() * expected) << id;
    }
}

TEST(SphCentralDifferenceOperators, ExtremeStepRatiosPreserveRepresentableDensityIncrements) {
    using sph::central_difference_step::TimeScheme3D;
    const Kokkos::Serial exec;
    const auto density = initial_density({1});
    sph::Values rate("test.density_rate", 1), previous_rate("test.previous_rate", 1);
    sph::Values next_density("test.next_density", 1);
    const auto advance = [&](real dt, real previous_dt) {
        return sph::central_difference_step::advance_density(exec, {}, {}, density, rate, {}, previous_rate,
            {TimeScheme3D::adams_bashforth, dt, previous_dt}, next_density);
    };

    const real tiny = std::numeric_limits<real>::denorm_min();
    rate(0) = tiny;
    previous_rate(0) = 0;
    ASSERT_TRUE(advance(.5, 2 * tiny).ok());
    EXPECT_EQ(next_density(0), real{1.0625});

    const real dt = std::ldexp(real{1}, 8 - std::numeric_limits<real>::max_exponent);
    rate(0) = -1 / dt;
    previous_rate(0) = 2 * rate(0);
    // The Euler contribution cancels density, and the remaining increment is dt.
    ASSERT_TRUE(advance(dt, .5).ok());
    EXPECT_EQ(next_density(0), dt);
}

TEST(SphContinuity, StationaryUniformDensityIsIndependentOfKernelMassSum) {
    Fields fields({{{0, 0, 0}, 2, 31, 0}, {{.5, 0, 0}, 3, 19, 0}});
    sph::Velocity3D velocity("continuity_test.velocity", 2);
    auto rho = initial_density({7, 7});
    sph::WcsphParticipant3D participant(fields.get(), 16, {{7, 1, 3}, 1}, velocity, rho);
    Simulation sim(participant);
    ASSERT_TRUE(sim.initialize().ok());
    rho(0) = 5;

    ASSERT_TRUE(sim.run(.25, 1).ok());
    const auto rows = committed_rows(participant);
    ASSERT_EQ(rows.size(), 2u);
    for (const auto& row : rows) {
        EXPECT_EQ(row.density, 7);
        EXPECT_EQ(row.pressure, 0);
        EXPECT_EQ(row.velocity, (std::array<real, 3>{}));
        EXPECT_EQ(row.mass, row.id == 31 ? 2 : 3);
    }
    EXPECT_TRUE(sim.finalize().ok());
}

TEST(SphContinuity, FixedWallCompressionAndExpansionFollowAnalyticRelativeVelocityGradient) {
    // h=2 keeps both endpoints on the interior branch of the M4 spline.
    // Startup uses the initial relative velocity and the initial kernel gradient.
    Fields fluid({{{.5, 0, 0}, 2, 42, 0}}), wall({{{1.5, 0, 0}, 3, 15, 1}});
    constexpr double h = 2, dt = .25;
    const sph::WcsphVerletParameters parameters{{2, 1, 1}, 1, {}, h};
    const sph::FixedWallParticles3D boundary{wall.position, wall.mass, wall.ids};
    for (real speed : {real{-.5}, real{0}, real{.5}}) {
        sph::Velocity3D velocity("continuity_test.velocity", 1);
        velocity(0, 0) = speed;
        sph::WcsphParticipant3D participant(fluid.get(), 16, parameters, velocity,
                                            initial_density({2}), boundary);
        Simulation sim(participant);
        ASSERT_TRUE(sim.initialize().ok());

        ASSERT_TRUE(sim.run(dt, 1).ok());
        const auto rate = [speed](double separation) {
            const double q = separation / h;
            return 3 * speed * (3 * q - 2.25 * q * q) / (std::numbers::pi * h * h * h * h);
        };
        const auto rows = committed_rows(participant);
        ASSERT_EQ(rows.size(), 1u);
        const double change = dt * rate(1);
        EXPECT_NEAR(rows[0].density, 2 + change, 4 * std::numeric_limits<real>::epsilon());
        if (speed != 0)
            EXPECT_GT((rows[0].density - 2) * speed, 0);
        EXPECT_EQ(rows[0].mass, 2);
        EXPECT_TRUE(sim.finalize().ok());
    }
}

long double density_invariant_error(int steps) {
    // The continuity ODE implies rho_i - sum_j m_j W_ij is constant for any
    // trajectory. Both initial densities deliberately differ from the kernel sum.
    std::vector<Particle> cloud{{{1, 0, 0}, 3, 46, 0}, {{0, 0, 0}, 2, 13, 0}};
    Fields fields(cloud);
    sph::Velocity3D velocity("continuity_test.velocity", 2);
    velocity(0, 0) = .75;
    velocity(1, 0) = -.25;
    sph::WcsphParticipant3D participant(fields.get(), 32, {{2, real{.01}, 1}, 1}, velocity,
                                        initial_density({2, 3}));
    Simulation sim(participant);
    EXPECT_TRUE(sim.initialize().ok());

    const auto initial = explicit_csr(cloud);
    EXPECT_TRUE(sim.run(.75 / steps, steps).ok());
    const auto rows = committed_rows(participant);
    EXPECT_EQ(rows.size(), cloud.size());
    for (const auto& row : rows) {
        const auto i = row.id == 46 ? 0u : 1u;
        cloud[i].position = row.position;
        EXPECT_EQ(row.mass, cloud[i].mass);
    }
    const auto final = explicit_csr(cloud);
    long double error{};
    for (const auto& row : rows) {
        const auto exact_change = final.density.at(row.id) - initial.density.at(row.id);
        EXPECT_GT(std::abs(exact_change), .1L);
        error = std::max(error, std::abs(row.density - (row.id == 46 ? 2 : 3) - exact_change));
    }
    EXPECT_TRUE(sim.finalize().ok());
    return error;
}

TEST(SphContinuity, SmoothCoupledPairHasSecondOrderDensityError) {
    const auto coarse = density_invariant_error(2);
    const auto medium = density_invariant_error(4);
    const auto fine = density_invariant_error(8);
    const auto record = [](const char* key, long double value) {
        std::ostringstream text;
        text << std::setprecision(16) << value;
        ::testing::Test::RecordProperty(key, text.str());
    };
    record("density_error_dt_0_375", coarse);
    record("density_error_dt_0_1875", medium);
    record("density_error_dt_0_09375", fine);
    record("observed_order_coarse_medium", std::log2(coarse / medium));
    record("observed_order_medium_fine", std::log2(medium / fine));
    EXPECT_GT(fine, 100 * std::numeric_limits<real>::epsilon());
    EXPECT_NEAR(coarse / medium, 4, .15);
    EXPECT_NEAR(medium / fine, 4, .15);
}

TEST(SphContinuity, CrossingParticlesPreserveUnequalMassAndDensityById) {
    Fields fields({{{2.25, 0, 0}, 2, 73, 0}, {{1.75, 0, 0}, 5, 19, 0}});
    sph::Velocity3D velocity("continuity_test.velocity", 2);
    velocity(0, 0) = -1;
    velocity(1, 0) = 1;
    const sph::WcsphVerletParameters parameters{{2, real{.001}, 1}, 1};
    sph::WcsphParticipant3D participant(fields.get(), 16, parameters, velocity, initial_density({2, 8}));
    Simulation sim(participant);
    ASSERT_TRUE(sim.initialize().ok());

    const auto before = committed_rows(participant);
    ASSERT_TRUE(sim.run(.5, 1).ok());
    const auto after = committed_rows(participant);
    ASSERT_EQ(before.size(), 2u);
    ASSERT_EQ(after.size(), 2u);
    EXPECT_NE(before[0].id, after[0].id);
    for (const auto& row : after) {
        const bool first = row.id == 73;
        EXPECT_EQ(row.mass, first ? 2 : 5);
        // Both particles approach with relative speed 2 at separation 1/2.
        const double gradient = (.5 * 3 - .25 * 2.25) / std::numbers::pi;
        const double increment = .5 * 2 * (first ? 5 : 2) * gradient;
        EXPECT_NEAR(row.density, (first ? 2 : 8) + increment, 1.e-6);
    }
    EXPECT_TRUE(sim.finalize().ok());
}

TEST(SphContinuity, NonpositiveCandidateDensityPreservesAcceptedRows) {
    constexpr real initial = .125, mass = 1, speed = 1;
    constexpr double failed_dt = .5;
    Fields fields({{{.5, 0, 0}, mass, 21, 0}, {{0, 0, 0}, mass, 42, 0}});
    sph::Velocity3D velocity("continuity_test.velocity", 2);
    velocity(0, 0) = speed;
    velocity(1, 0) = -speed;
    const sph::WcsphVerletParameters parameters{{initial, real{.125}, 1}, 1};
    const auto rho = initial_density({initial, initial});
    sph::WcsphParticipant3D participant(fields.get(), 32, parameters, velocity, rho);
    Simulation sim(participant);
    ASSERT_TRUE(sim.initialize().ok());


    const auto before = committed_rows(participant);
    ASSERT_EQ(before.size(), 2u);
    const auto bound = participant.time_constraint().maximum_step;
    ASSERT_LE(failed_dt, bound);
    const auto failure = sim.run(failed_dt, 1);
    EXPECT_EQ(failure.code(), StatusCode::numerical_failure);
    EXPECT_NE(failure.message().find("owned WCSPH density"), std::string::npos)
        << failure.message();
    EXPECT_EQ(sim.time().accepted_steps, 0u);
    EXPECT_EQ(sim.time().accepted_time, 0);
    EXPECT_EQ(participant.time_constraint().maximum_step, bound);
    const auto restored = committed_rows(participant);
    ASSERT_EQ(restored.size(), before.size());
    for (std::size_t row = 0; row < restored.size(); ++row) {
        SCOPED_TRACE(before[row].id);
        EXPECT_EQ(restored[row], before[row]);
        EXPECT_EQ(restored[row].density, initial);
    }
    EXPECT_TRUE(sim.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

} // namespace
