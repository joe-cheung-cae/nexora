#include "cell_grid_oracle.hpp"

#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <nexora/simulation/Simulation.hpp>
#include <participants/sph/WcsphParticipant3D.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace nexora;
namespace wall = sph::SphBoundary;

constexpr double end_time = .5;
constexpr double wall_body_mass = 1.5, follower_mass = 1., stiffness = 6.;
constexpr real initial_fluid_velocity = -.5;
constexpr real wall_reference_x = -1;

struct Sliders {
    double wall_x{}, follower_x{}, wall_v{}, follower_v{}, wall_a{};
};

// Two frictionless x-sliders linked by an initially relaxed spring. Positions
// are displacements in m; positive forces act along world +x. No gravity.
using MotionTable = Kokkos::View<Sliders*, Kokkos::Serial>;
void solve_sliders(const Kokkos::Serial& exec, const MotionTable& samples,
                   double dt, double wall_load) {
    const int steps = samples.extent_int(0) - 1;
    Kokkos::parallel_for("test.two_sliders.velocity_verlet", Kokkos::RangePolicy<Kokkos::Serial>(exec, 0, 1),
        KOKKOS_LAMBDA(int) {
            auto state = samples(0);
            for (int step = 1; step <= steps; ++step) {
                double spring = stiffness * (state.follower_x - state.wall_x);
                state.wall_v += .5 * dt * (wall_load + spring) / wall_body_mass;
                state.follower_v -= .5 * dt * spring / follower_mass;
                state.wall_x += dt * state.wall_v;
                state.follower_x += dt * state.follower_v;
                spring = stiffness * (state.follower_x - state.wall_x);
                state.wall_a = (wall_load + spring) / wall_body_mass;
                state.wall_v += .5 * dt * state.wall_a;
                state.follower_v -= .5 * dt * spring / follower_mass;
                samples(step) = state;
            }
        });
    exec.fence("publish slider samples to Serial wall callback");
}

void require_ok(const Status& status) {
    if (!status.ok())
        throw std::runtime_error(status.message());
}

sph::CommittedRow3D fluid_row(const sph::WcsphParticipant3D& fluid) {
    sph::CommittedRow3D result{};
    require_ok(fluid.visit_committed_rows([&](auto row) { result = row; }));
    EXPECT_EQ(result.id, 11u);
    return result;
}

struct ExchangeResult {
    Sliders sliders;
    double wall_impulse, fluid_impulse, integrated_pressure_force;
};

ExchangeResult run_exchange(int windows, int total_substeps, int slider_stride = 1, bool feedback = true) {
    SCOPED_TRACE(::testing::Message() << windows << " windows, " << total_substeps << " SPH steps, "
                 << slider_stride << " mechanical steps per SPH step, feedback=" << feedback);
    const int substeps = total_substeps / windows;
    const double dt = end_time / total_substeps;
    const int slider_substeps = substeps * slider_stride;
    const double slider_dt = dt / slider_stride;
    const Kokkos::Serial exec;
    const MotionTable samples(Kokkos::view_alloc(exec, "test.slider_motion"), slider_substeps + 1);
    double window_start = 0.;
    // Both clocks are dyadic. SPH samples only physical substep endpoints in
    // this window, including sample 0 during initialization.
    const wall::PrescribedWallTrajectory3D trajectory{
        .motion = [&](wall::WallTrajectoryTime3D time, double) -> Result<wall::RigidWallMotion3D> {
            const double index = (time.elapsed - window_start) / slider_dt;
            if (index < 0 || index > slider_substeps || index != std::floor(index))
                return Status{StatusCode::invalid_state, "time outside slider motion samples"};
            const auto& s = samples(static_cast<int>(index));
            return wall::RigidWallMotion3D{.translation = {static_cast<real>(s.wall_x), 0, 0},
                                           .velocity = {static_cast<real>(s.wall_v), 0, 0},
                                           .acceleration = {static_cast<real>(s.wall_a), 0, 0}};
        },
        .sample_step_limit = dt};
    const sph_test::Fields fluid({{{0, 0, 0}, .75, 11, 0}}), boundary({{{wall_reference_x, 0, 0}, 1, 29, 1}});
    const sph::Velocity3D velocity("test.exchange.fluid_velocity", 1);
    velocity(0, 0) = initial_fluid_velocity;
    const sph::Values density("test.exchange.fluid_density", 1);
    density(0) = 1;
    const std::vector<wall::WallGroupInput3D> groups{
        {21, {boundary.position, boundary.mass, boundary.ids}, trajectory}};
    sph::WcsphParticipant3D participant(fluid.get(), 64, {{1, 2, 1}, 1}, velocity, density, groups);
    Simulation simulation(participant);
    require_ok(simulation.initialize());
    auto row = fluid_row(participant);
    EXPECT_EQ(row.pressure_acceleration[0], 0); // relaxed spring, zero initial pressure
    double impulse = 0., integrated_pressure_force = 0.;
    for (int n = 0; n < windows; ++n) {
        window_start = (n * substeps) * dt;
        // At t_n read the accepted SPH reaction (N). With exactly one fluid
        // particle, one wall and zero body force, the entire pressure force is
        // from this wall. Hold its negative on the wall slider until t_(n+1).
        const double force = feedback ? -double(row.mass) * row.pressure_acceleration[0] : 0.;
        impulse += force * (substeps * dt);
        if (n > 0)
            samples(0) = samples(slider_substeps); // left-limit acceleration, continuous x and v
        solve_sliders(exec, samples, slider_dt, force);
        // Exchange the completed motion table once. SPH retains the accepted
        // t_n acceleration from the preceding window. Every new Adami endpoint
        // uses the acceleration at its pose under this window's held load.
        for (int i = 0; i < substeps; ++i) {
            const double previous_force = double(row.mass) * row.pressure_acceleration[0];
            require_ok(simulation.run(dt));
            row = fluid_row(participant);
            const double endpoint_force = double(row.mass) * row.pressure_acceleration[0];
            integrated_pressure_force += .5 * dt * (previous_force + endpoint_force);
        }
        const auto wall_end = sph_test::checked(participant.read_committed_wall(29));
        const double window_end = (n + 1) * substeps * dt;
        EXPECT_EQ(simulation.time().accepted_time, window_end);
        EXPECT_EQ(wall_end.stamp.time.physical_time, window_end);
        EXPECT_EQ(wall_end.position[0], wall_reference_x + static_cast<real>(samples(slider_substeps).wall_x));
    }
    const auto final = samples(slider_substeps);
    // The spring sums to zero across both bodies, so their total momentum
    // independently measures the impulse actually applied to the mechanism.
    EXPECT_NEAR(wall_body_mass * final.wall_v + follower_mass * final.follower_v, impulse, 1.e-12);
    require_ok(simulation.finalize());
    return {final, impulse, double(row.mass) * (row.velocity[0] - initial_fluid_velocity), integrated_pressure_force};
}

TEST(SphMbdExchange, TwoSlidersRefinePressureLoadAndReturnedMotion) {
    constexpr int internal_steps = 256;
    std::array<ExchangeResult, 4> runs{};
    for (int level = 0; level < 4; ++level) {
        const int windows = 4 << level;
        runs[level] = run_exchange(windows, internal_steps);
        const auto& r = runs[level];
        std::ostringstream report;
        report << std::setprecision(16) << "wall_x=" << r.sliders.wall_x << " follower_x=" << r.sliders.follower_x
               << " wall_J=" << r.wall_impulse << " fluid_J=" << r.fluid_impulse
               << " pressure_trapezoid=" << r.integrated_pressure_force;
        ::testing::Test::RecordProperty("windows_" + std::to_string(windows), report.str());
        EXPECT_LT(r.sliders.wall_x, 0);
        EXPECT_LT(r.sliders.follower_x, 0);
        EXPECT_LT(r.wall_impulse, 0);
        EXPECT_GT(r.fluid_impulse, 0);
    }
    const auto fine_sliders = run_exchange(32, internal_steps, 2);
    const auto fine_both = run_exchange(32, 2 * internal_steps);
    const auto& finest = runs.back();
    const auto check_refinement = [&](const char* name, auto quantity) {
        SCOPED_TRACE(name);
        double previous_difference{};
        for (int i = 1; i < 4; ++i) {
            const double difference = std::abs(quantity(runs[i]) - quantity(runs[i - 1]));
            EXPECT_GT(difference, 0.);
            if (i > 1)
                EXPECT_GT(previous_difference, 1.4 * difference);
            previous_difference = difference;
        }
        const double slider_difference = std::abs(quantity(fine_sliders) - quantity(finest));
        const double fluid_difference = std::abs(quantity(fine_both) - quantity(fine_sliders));
        EXPECT_LT(slider_difference, .1 * previous_difference);
        EXPECT_LT(fluid_difference, .1 * previous_difference);
        std::ostringstream report;
        report << std::setprecision(16) << "sliders=" << slider_difference << " SPH=" << fluid_difference;
        ::testing::Test::RecordProperty(std::string("internal_difference_") + name, report.str());
    };
    check_refinement("wall_x", [](const auto& r) { return r.sliders.wall_x; });
    check_refinement("follower_x", [](const auto& r) { return r.sliders.follower_x; });
    check_refinement("wall_J", [](const auto& r) { return r.wall_impulse; });
    check_refinement("fluid_J", [](const auto& r) { return r.fluid_impulse; });
    check_refinement("pressure_trapezoid", [](const auto& r) { return r.integrated_pressure_force; });
    for (int i = 1; i < 4; ++i)
        EXPECT_LT(std::abs(runs[i].wall_impulse + runs[i].fluid_impulse),
                  std::abs(runs[i - 1].wall_impulse + runs[i - 1].fluid_impulse));
    const auto disconnected = run_exchange(32, internal_steps, 1, false);
    ::testing::Test::RecordProperty("disconnected_fluid_J", std::to_string(disconnected.fluid_impulse));
    EXPECT_EQ(disconnected.sliders.wall_x, 0);
    EXPECT_EQ(disconnected.sliders.follower_x, 0);
    EXPECT_GT(std::abs(finest.fluid_impulse - disconnected.fluid_impulse), 1.e-5);
}
} // namespace
