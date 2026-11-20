#include "cell_grid_oracle.hpp"
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <nexora/simulation/Simulation.hpp>
#include <numbers>
#include <participants/sph/Participant3D.hpp>
#include <sstream>

namespace {
using namespace nexora;
using namespace sph_test;

sph::Values initial_density(std::initializer_list<real> entries) {
    sph::Values result("continuity_test.density", entries.size());
    std::copy(entries.begin(), entries.end(), result.data());
    return result;
}

TEST(SphContinuity, StationaryUniformDensityIsIndependentOfKernelMassSum) {
    Fields fields({{{0, 0, 0}, 2, 31, 0}, {{.5, 0, 0}, 3, 19, 0}});
    sph::Velocity3D velocity("continuity_test.velocity", 2);
    auto rho = initial_density({7, 7});
    sph::Participant3D participant(fields.get(), 16, {{7, 1, 3}, 1}, velocity, rho);
    Simulation sim(participant);
    ASSERT_TRUE(sim.initialize().ok());
    rho(0) = 5;
    ASSERT_TRUE(sim.prepare().ok());
    ASSERT_TRUE(sim.advance(.25).ok());
    const auto state = checked(participant.snapshot());
    for (std::size_t row = 0; row < state.row_count(); ++row) {
        EXPECT_EQ(state.density(row), 7);
        EXPECT_EQ(checked(state.pressure(row)), 0);
        EXPECT_EQ(checked(state.velocity(row)), (std::array<real, 3>{}));
        EXPECT_EQ(state.mass(row), state.row_id(row) == 31 ? 2 : 3);
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
        sph::Participant3D participant(fluid.get(), 16, parameters, velocity, initial_density({2}),
                                       boundary);
        Simulation sim(participant);
        ASSERT_TRUE(sim.initialize().ok());
        ASSERT_TRUE(sim.prepare().ok());
        ASSERT_TRUE(sim.advance(dt).ok());
        const auto rate = [speed](double separation) {
            const double q = separation / h;
            return 3 * speed * (3 * q - 2.25 * q * q) / (std::numbers::pi * h * h * h * h);
        };
        const auto state = checked(participant.snapshot());
        const double change = dt * .5 * (rate(1) + rate(1 - speed * dt));
        EXPECT_NEAR(state.density(0), 2 + change, 4 * std::numeric_limits<real>::epsilon());
        if (speed != 0)
            EXPECT_GT((state.density(0) - 2) * speed, 0);
        EXPECT_EQ(state.row_count(), 1u);
        EXPECT_EQ(state.mass(0), 2);
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
    sph::Participant3D participant(fields.get(), 32, {{2, real{.01}, 1}, 1}, velocity,
                                   initial_density({2, 3}));
    Simulation sim(participant);
    EXPECT_TRUE(sim.initialize().ok());
    EXPECT_TRUE(sim.prepare().ok());
    const auto initial = explicit_csr(cloud);
    for (int n = 0; n < steps; ++n)
        EXPECT_TRUE(sim.advance(.75 / steps).ok());
    const auto state = checked(participant.snapshot());
    for (std::size_t row = 0; row < state.row_count(); ++row) {
        const auto i = state.row_id(row) == 46 ? 0u : 1u;
        cloud[i].position = checked(state.position(row));
        EXPECT_EQ(state.mass(row), cloud[i].mass);
    }
    const auto final = explicit_csr(cloud);
    long double error{};
    for (std::size_t row = 0; row < state.row_count(); ++row) {
        const auto id = state.row_id(row);
        const auto exact_change = final.density.at(id) - initial.density.at(id);
        EXPECT_GT(std::abs(exact_change), .1L);
        error = std::max(error, std::abs(state.density(row) - (id == 46 ? 2 : 3) - exact_change));
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
    sph::Participant3D participant(fields.get(), 16, parameters, velocity, initial_density({2, 8}));
    Simulation sim(participant);
    ASSERT_TRUE(sim.initialize().ok());
    ASSERT_TRUE(sim.prepare().ok());
    const auto before = checked(participant.snapshot());
    ASSERT_TRUE(sim.advance(.5).ok());
    const auto after = checked(participant.snapshot());
    EXPECT_NE(before.row_id(0), after.row_id(0));
    for (std::size_t row = 0; row < after.row_count(); ++row) {
        const bool first = after.row_id(row) == 73;
        EXPECT_EQ(after.mass(row), first ? 2 : 5);
        // Endpoints are mirrored around the crossing in the ballistic limit,
        // so the endpoint rates cancel to within the small pressure correction.
        EXPECT_NEAR(after.density(row), first ? 2 : 8, 1.e-5);
    }
    EXPECT_TRUE(sim.finalize().ok());
}

TEST(SphContinuity, InvalidExplicitDensityAndExtentPublishNoSnapshot) {
    Fields fields({{{0, 0, 0}, 1, 31, 0}});
    sph::Velocity3D velocity("continuity_test.velocity", 1);
    for (const auto& rho :
         {initial_density({}), initial_density({1, 1}), initial_density({0}), initial_density({-1}),
          initial_density({std::numeric_limits<real>::infinity()}),
          initial_density({std::numeric_limits<real>::quiet_NaN()})}) {
        sph::Participant3D participant(fields.get(), 16, {{1, 1, 1}, 1}, velocity, rho);
        Simulation sim(participant);
        EXPECT_FALSE(sim.initialize().ok());
        EXPECT_FALSE(participant.snapshot().ok());
        EXPECT_TRUE(sim.finalize().ok());
    }
}

void check_density_failure(real initial, real mass, real speed, double failed_dt, double retry_dt) {
    Fields fields({{{.5, 0, 0}, mass, 21, 0}, {{0, 0, 0}, mass, 42, 0}});
    sph::Velocity3D velocity("continuity_test.velocity", 2);
    velocity(0, 0) = speed;
    velocity(1, 0) = -speed;
    const sph::WcsphVerletParameters parameters{{initial, real{.125}, 1}, 1};
    const auto rho = initial_density({initial, initial});
    sph::Participant3D participant(fields.get(), 32, parameters, velocity, rho);
    sph::Participant3D clean(fields.get(), 32, parameters, velocity, rho);
    Simulation sim(participant), clean_sim(clean);
    ASSERT_TRUE(sim.initialize().ok());
    ASSERT_TRUE(sim.prepare().ok());
    ASSERT_TRUE(clean_sim.initialize().ok());
    ASSERT_TRUE(clean_sim.prepare().ok());
    const auto before = checked(participant.snapshot());
    ASSERT_LE(failed_dt, participant.time_constraint(0).maximum_step);
    const auto failure = sim.advance(failed_dt);
    EXPECT_EQ(failure.code(), StatusCode::numerical_failure);
    EXPECT_NE(failure.message().find("corrected WCSPH density"), std::string::npos)
        << failure.message();
    EXPECT_EQ(sim.time().accepted_steps, 0u);
    const auto restored = checked(participant.snapshot());
    EXPECT_EQ(before.stamp, restored.stamp);
    for (std::size_t row = 0; row < restored.row_count(); ++row) {
        EXPECT_EQ(restored.density(row), initial);
        EXPECT_EQ(checked(restored.position(row)), checked(before.position(row)));
        EXPECT_EQ(checked(restored.velocity(row)), checked(before.velocity(row)));
    }
    ASSERT_TRUE(sim.advance(retry_dt).ok());
    ASSERT_TRUE(clean_sim.advance(retry_dt).ok());
    const auto retried = checked(participant.snapshot()), reference = checked(clean.snapshot());
    for (std::size_t row = 0; row < retried.row_count(); ++row) {
        EXPECT_EQ(retried.row_id(row), reference.row_id(row));
        EXPECT_EQ(retried.density(row), reference.density(row));
        EXPECT_EQ(checked(retried.velocity(row)), checked(reference.velocity(row)));
        EXPECT_EQ(checked(retried.pressure(row)), checked(reference.pressure(row)));
    }
    EXPECT_TRUE(sim.finalize().ok());
    EXPECT_TRUE(clean_sim.finalize().ok());
}

TEST(SphContinuity, NonpositiveAndOverflowingTrialDensityRestoreAndRetry) {
    check_density_failure(real{.125}, 1, 1, .5, .03125);
    const real limit = std::numeric_limits<real>::max();
    check_density_failure(real{.95} * limit, real{.75} * limit, -1, .25, .015625);
}

} // namespace
