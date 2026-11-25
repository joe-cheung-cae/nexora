#include "cell_grid_oracle.hpp"
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <methods/sph/Verlet3D.hpp>
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

TEST(SphVerletOperators, CflReductionIncludesNonfiniteVelocityAmongFiniteRows) {
    Kokkos::Serial exec;
    sph::Velocity3D velocity("test.cfl_velocity", 2);
    velocity(0, 1) = std::numeric_limits<real>::quiet_NaN();
    velocity(1, 2) = 1;
    EXPECT_EQ(sph::cfl_step_3d(exec, velocity, 1, 1).status().code(),
              StatusCode::numerical_failure);
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
