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
    for (const bool standalone : {true, false}) {
        Kokkos::deep_copy(exec, position, real{-19});
        Kokkos::deep_copy(exec, half, real{-17});
        const auto result = standalone
            ? sph::kick_drift_3d(exec, grid, velocity, acceleration, .25, position, half)
            : sph::verlet_step::kick_drift(exec, grid, velocity, acceleration, .25, position, half, {});
        ASSERT_TRUE(result.ok()) << result.message();
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_EQ(half(0, axis), expected_half[axis]);
            EXPECT_EQ(position(0, axis), expected_position[axis]);
            EXPECT_EQ(half(1, axis), real{-17});
            EXPECT_EQ(position(1, axis), real{-19});
        }
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

#if !NEXORA_ENABLE_DOUBLE_PRECISION
TEST(SphVerletOperators, FloatStepChecksDoubleDriftBeforeRoundingBackToFiniteStorage) {
    const Kokkos::Serial exec;
    const real largest = std::numeric_limits<real>::max();
    const Fields fields({{{largest, 0, 0}, 1, 11, 0}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 1, {}, 0x1p70));
    sph::Velocity3D velocity("test.kick.velocity", 1), acceleration("test.kick.acceleration", 1);
    sph::Velocity3D position("test.kick.position", 1), half("test.kick.half", 1);
    const double exact_drift = double(largest) + 0x1p99;
    ASSERT_GT(exact_drift, double(largest));
    ASSERT_EQ(static_cast<real>(exact_drift), largest);
    EXPECT_EQ(sph::verlet_step::kick_drift(exec, grid, velocity, acceleration, 1,
                                           position, half, {0x1p100, 0, 0}).code(),
              StatusCode::numerical_failure);
}
#else
TEST(SphVerletOperators, DoubleStepRejectsKickOrDriftOverflowAtGridBoundary) {
    const Kokkos::Serial exec;
    const Fields fields({{{0, 0, 0}, 1, 11, 0}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 1));
    sph::Velocity3D velocity("test.kick.velocity", 1), acceleration("test.kick.acceleration", 1);
    sph::Velocity3D position("test.kick.position", 1), half("test.kick.half", 1);
    for (const bool overflow_half : {true, false}) {
        Kokkos::deep_copy(exec, velocity, real{0});
        Kokkos::deep_copy(exec, acceleration, real{0});
        velocity(0, 0) = overflow_half ? 0 : std::numeric_limits<real>::max() / 4;
        acceleration(0, 0) = overflow_half ? std::numeric_limits<real>::max() : 0;
        const double dt = overflow_half ? 4 : 8;
        ASSERT_TRUE(sph::verlet_step::kick_drift(exec, grid, velocity, acceleration, dt,
                                                position, half, {}).ok());
        EXPECT_FALSE(std::isfinite(position(0, 0)));
        EXPECT_EQ(std::isfinite(half(0, 0)), !overflow_half);
        EXPECT_EQ(grid.update_positions(exec, position, 1).status().code(),
                  StatusCode::invalid_argument);
        EXPECT_EQ(grid.fields().position(0, 0), 0);
    }
}
#endif

TEST(SphVerletOperators, CflReductionIncludesNonfiniteVelocityAmongFiniteRows) {
    Kokkos::Serial exec;
    sph::Velocity3D velocity("test.cfl_velocity", 2);
    velocity(0, 1) = std::numeric_limits<real>::quiet_NaN();
    velocity(1, 2) = 1;
    EXPECT_EQ(sph::cfl_step_3d(exec, velocity, 1, 1).status().code(),
              StatusCode::numerical_failure);
}

TEST(SphVerletOperators, FinalKickRejectsUnrepresentableVelocityAtWrite) {
    Kokkos::Serial exec;
    Fields fields({{{0, 0, 0}, 1, 31, 0}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 1));
    sph::Velocity3D half("test.half", 1), velocity("test.final_velocity", 1);
    sph::Acceleration3D acceleration("test.acceleration", 1);
    half(0, 0) = acceleration(0, 0) = std::numeric_limits<real>::max();
    velocity(0, 0) = 7;
    EXPECT_EQ(sph::finish_kick_3d(exec, grid, half, acceleration, 2, velocity).code(),
              StatusCode::numerical_failure);
    EXPECT_EQ(velocity(0, 0), 7);
}

TEST(SphContinuity, BothStagesRejectNonfiniteRateAtDensityCorrection) {
    Kokkos::Serial exec;
    Fields fields({{{0, 0, 0}, 1, 31, 0}, {{1, 0, 0}, 1, 42, 0}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 2));
    const auto& query = grid.neighbors();
    sph::DensityRate3D start("test.start_rate", 2), end("test.end_rate", 2);
    sph::Velocity3D velocity("test.velocity", 2);
    const auto density = initial_density({1, 1});
    sph::Values corrected("test.corrected", 2);
    velocity(0, 0) = std::numeric_limits<real>::quiet_NaN();
    EXPECT_EQ(sph::continuity_rate_3d(exec, query, velocity, start).code(),
              StatusCode::numerical_failure);
    for (const bool at_start : {true, false}) {
        auto& invalid = at_start ? start : end;
        auto& valid = at_start ? end : start;
        Kokkos::deep_copy(exec, valid, 0.);
        sph::stage_continuity_rate_3d(exec, query, velocity, invalid);
        // Both rows and both slot permutations are the identity for this grid.
        const auto status = sph::correct_density_3d(exec, grid.owned_slots(), grid.sorted_to_input(),
            grid.input_to_sorted(), density, start, end, .25, corrected);
        EXPECT_EQ(status.code(), StatusCode::numerical_failure);
    }
}

TEST(SphContinuity, PredictionReadsOnlyActualGeometricContributors) {
    Kokkos::Serial exec;
    const real tiny = std::numeric_limits<real>::denorm_min();
    Fields fields({{{0, 0, 0}, 1, 10, 0}, {{0, 0, 0}, 1, 11, 1}, {{2, 0, 0}, 1, 12, 1},
                   {{10, 0, 0}, 1, 20, 0}, {{10.5, 0, 0}, 1, 21, 0}, {{11, 0, 0}, 1, 22, 1},
                   {{20, 0, 0}, 1, 30, 0}, {{0, 10, 0}, 1, 40, 0}, {{tiny, 10, 0}, 1, 41, 1}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 128));
    const auto& query = grid.neighbors();
    const auto& ids = query.fields().ids;
    const auto& owned = query.owned_slots();
    sph::Velocity3D velocity("test.predicted_velocity", ids.extent(0));
    Kokkos::deep_copy(exec, velocity, real{77});
    sph::Velocity3D accepted("test.accepted_velocity", owned.extent(0));
    sph::Acceleration3D acceleration("test.accepted_acceleration", owned.extent(0));
    sph::NeighborView3D::Indices::non_const_type old_row("test.old_rows", ids.extent(0));
    for (std::size_t row = 0; row < owned.extent(0); ++row) {
        const auto slot = owned(row);
        const auto id = ids(slot);
        old_row(grid.sorted_to_input()(slot)) = static_cast<LocalIndex>(row);
        accepted(row, 0) = id == 20 ? real{1} : real{0};
        if (id == 10 || id == 30)
            accepted(row, 0) = std::numeric_limits<real>::quiet_NaN();
    }
    const sph::Velocity3D::const_type read_velocity = accepted;
    const sph::Acceleration3D::const_type read_acceleration = acceleration;
    const std::array<double, 3> gravity{};
    const sph::ContinuityPredictor3D predictor{read_velocity, read_acceleration, .25, gravity};
    sph::StageVelocityReady3D ready("test.predicted_velocity_ready", ids.extent(0));
    sph::DensityRate3D rate("test.predicted_rate", owned.extent(0));
    std::vector<std::uint64_t> read_walls;
    const auto write = [&](LocalIndex slot) {
        const auto id = ids(slot);
        read_walls.push_back(id);
        if (id != 22 && id != 41)
            return false;
        velocity(slot, 0) = velocity(slot, 1) = velocity(slot, 2) = real{0};
        return true;
    };
    ASSERT_TRUE(sph::stage_predicted_continuity_rate_3d(exec, query, velocity, rate, ready,
        predictor, grid.sorted_to_input(), old_row, std::cref(write)).ok());
    // The last nonzero displacement can square to zero in double and still
    // consumes both velocities. Every contributing slot is materialized once.
    std::sort(read_walls.begin(), read_walls.end());
    EXPECT_EQ(read_walls, (std::vector<std::uint64_t>{22, 41}));
    for (std::size_t row = 0; row < rate.extent(0); ++row) {
        const auto slot = owned(row);
        const auto id = ids(slot);
        if (id == 10 || id == 30)
            EXPECT_EQ(velocity(slot, 0), 77);
        else
            EXPECT_EQ(velocity(slot, 0), id == 20 ? 1 : 0);
        if (id == 20 || id == 21)
            EXPECT_GT(rate(row), 0);
        else
            EXPECT_EQ(rate(row), 0);
    }
}

#if NEXORA_ENABLE_DOUBLE_PRECISION
TEST(SphContinuity, RoundedSubnormalSupportTieStillConsumesVelocity) {
    Kokkos::Serial exec;
    constexpr double h = 0x1.2p-538, separation = 0x1p-537;
    const double support = 2 * h;
    ASSERT_EQ(separation * separation, support * support);
    ASSERT_LT(std::sqrt(separation * separation) / h, 2);
    Fields fields({{{0, 0, 0}, 1, 10, 0}, {{separation, 0, 0}, 1, 11, 1}});
    const auto grid = checked(sph::CellGrid::build(exec, fields.get(), 1, {}, h));
    sph::Velocity3D accepted("test.accepted_velocity", 1), scratch("test.predicted_velocity", 2);
    sph::Acceleration3D acceleration("test.accepted_acceleration", 1);
    sph::StageVelocityReady3D ready("test.prediction_ready", 2);
    sph::DensityRate3D start("test.start_rate", 1), end("test.end_rate", 1);
    accepted(0, 0) = 1;
    sph::NeighborView3D::Indices::non_const_type old_row("test.old_row", 2);
    const sph::Velocity3D::const_type read_velocity = accepted;
    const sph::Acceleration3D::const_type read_acceleration = acceleration;
    const std::array<double, 3> gravity{};
    const sph::ContinuityPredictor3D predictor{read_velocity, read_acceleration, .25, gravity};
    std::vector<std::uint64_t> wall_reads;
    const auto write = [&](LocalIndex slot) {
        wall_reads.push_back(grid.fields().ids(slot));
        return true; // Scratch initializes this wall's velocity to zero.
    };
    ASSERT_TRUE(sph::stage_predicted_continuity_rate_3d(exec, grid.neighbors(), scratch, end,
        ready, predictor, grid.sorted_to_input(), old_row, std::cref(write)).ok());
    EXPECT_EQ(wall_reads, (std::vector<std::uint64_t>{11}));
    EXPECT_TRUE(std::isinf(end(0)));
    const auto density = initial_density({1});
    sph::Values corrected("test.corrected_density", 1);
    EXPECT_EQ(sph::correct_density_3d(exec, grid.owned_slots(), grid.sorted_to_input(), old_row,
        density, start, end, .25, corrected).code(), StatusCode::numerical_failure);
}
#endif

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
    // Zero initial pressure and gravity give an exactly ballistic predictor.
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
        const double change = dt * .5 * (rate(1) + rate(1 - speed * dt));
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
        // Endpoints are mirrored around the crossing in the ballistic limit,
        // so the endpoint rates cancel to within the small pressure correction.
        EXPECT_NEAR(row.density, first ? 2 : 8, 1.e-5);
    }
    EXPECT_TRUE(sim.finalize().ok());
}

TEST(SphContinuity, InvalidExplicitDensityAndExtentPublishNoRows) {
    Fields fields({{{0, 0, 0}, 1, 31, 0}});
    sph::Velocity3D velocity("continuity_test.velocity", 1);
    const auto rejects = [&](const sph::Values& rho, StatusCode reason) {
        sph::WcsphParticipant3D participant(fields.get(), 16, {{1, 1, 1}, 1}, velocity, rho);
        Simulation sim(participant);
        EXPECT_EQ(sim.initialize().code(), reason);
        std::size_t visits = 0;
        const auto status = participant.visit_committed_rows(
            [&](sph::CommittedRow3D) { ++visits; });
        EXPECT_EQ(status.code(), StatusCode::invalid_state);
        EXPECT_EQ(visits, 0u);
        EXPECT_TRUE(sim.finalize().ok());
    };
    rejects(initial_density({1, 1}), StatusCode::invalid_argument);
    for (const real density : {real{0}, std::numeric_limits<real>::infinity()}) {
        SCOPED_TRACE(density);
        rejects(initial_density({density}), StatusCode::numerical_failure);
    }
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
        EXPECT_EQ(restored[row].id, before[row].id);
        EXPECT_EQ(restored[row].mass, before[row].mass);
        EXPECT_EQ(restored[row].density, initial);
        EXPECT_EQ(restored[row].position, before[row].position);
        EXPECT_EQ(restored[row].velocity, before[row].velocity);
        EXPECT_EQ(restored[row].pressure, before[row].pressure);
        EXPECT_EQ(restored[row].pressure_acceleration, before[row].pressure_acceleration);
    }
    EXPECT_TRUE(sim.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
}

} // namespace
