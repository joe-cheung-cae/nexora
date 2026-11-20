#include "wcsph_oracle.hpp"
#include <iomanip>
#include <nexora/simulation/Simulation.hpp>
#include <sstream>

namespace {
using namespace nexora;
using namespace sph_test;
using Velocities = std::map<std::uint64_t, std::array<real, 3>>;
struct Trajectory {
    std::vector<Particle> particles;
    Velocities velocity;
    std::map<std::uint64_t, long double> density{};
};
sph::Velocity3D input_velocity(const Trajectory& state) {
    sph::Velocity3D result("test.initial_velocity", state.particles.size());
    for (std::size_t i = 0; i < state.particles.size(); ++i)
        for (int k = 0; k < 3; ++k)
            result(i, k) = state.velocity.at(state.particles[i].id)[k];
    return result;
}
sph::Values seed_density(const std::vector<Particle>& particles) {
    const auto sum = explicit_csr(particles);
    sph::Values density("test.initial_density", particles.size());
    for (std::size_t i = 0; i < particles.size(); ++i)
        density(i) = static_cast<real>(sum.density.at(particles[i].id));
    return density;
}
long double density_rate(const Particle& particle, const std::vector<Particle>& cloud,
                         const Velocities& velocity) {
    long double sum{};
    for (const auto& other : cloud) {
        std::array<long double, 3> delta;
        long double dot{};
        for (int k = 0; k < 3; ++k) {
            delta[k] = static_cast<long double>(particle.position[k]) - other.position[k];
            dot +=
                (static_cast<long double>(velocity.at(particle.id)[k]) - velocity.at(other.id)[k]) *
                delta[k];
        }
        const long double r = std::hypot(delta[0], delta[1], delta[2]);
        if (r > 0 && r < 2) {
            const auto radial = r < 1 ? (2.25L * r - 3) : -.75L * (2 - r) * (2 - r) / r;
            sum += other.mass * radial * dot / std::acos(-1.L);
        }
    }
    return sum;
}
Trajectory explicit_verlet(const Trajectory& accepted, sph::WcsphParameters eos, double dt) {
    // All forces originate in the independent all-pairs CSR oracle. Match the
    // public real state boundaries at each half-kick/drift/final-kick, not the
    // production cell ordering or numerical kernels.
    const auto old_force = force_oracle(accepted.particles, eos, accepted.density);
    Trajectory next = accepted;
    Velocities half, predictor;
    for (auto& particle : next.particles) {
        for (int k = 0; k < 3; ++k) {
            const real acceleration = static_cast<real>(old_force.acceleration.at(particle.id)[k]);
            predictor[particle.id][k] = static_cast<real>(
                std::fma(dt, double(acceleration), double(accepted.velocity.at(particle.id)[k])));
            half[particle.id][k] = static_cast<real>(std::fma(
                dt / 2, double(acceleration), double(accepted.velocity.at(particle.id)[k])));
            particle.position[k] = static_cast<real>(
                std::fma(dt, double(half.at(particle.id)[k]), double(particle.position[k])));
        }
    }
    for (std::size_t i = 0; i < next.particles.size(); ++i) {
        const auto id = next.particles[i].id;
        const auto before =
            density_rate(accepted.particles[i], accepted.particles, accepted.velocity);
        const auto after = density_rate(next.particles[i], next.particles, predictor);
        next.density[id] = static_cast<real>(static_cast<real>(old_force.csr.density.at(id)) +
                                             dt * .5L * (before + after));
    }
    const auto refreshed = force_oracle(next.particles, eos, next.density);
    for (const auto& particle : next.particles)
        for (int k = 0; k < 3; ++k)
            next.velocity[particle.id][k] = static_cast<real>(std::fma(
                dt / 2, double(static_cast<real>(refreshed.acceleration.at(particle.id)[k])),
                double(half.at(particle.id)[k])));
    return next;
}
struct TrajectoryErrors {
    long double position{}, velocity{}, density{}, pressure{};
    std::array<long double, 3> acceleration{};
};
void check_trajectory(const sph::Snapshot3D& snapshot, const Trajectory& expected,
                      sph::WcsphParameters eos, std::size_t steps,
                      TrajectoryErrors* maximum = nullptr) {
    // Check fields against the oracle at the observed geometry, avoiding force
    // cancellation bounds that pretend accumulated trajectory roundoff is zero.
    auto observed = expected.particles;
    const auto independent_force = force_oracle(expected.particles, eos, expected.density);
    const long double tolerance = 256 * (steps + 1) * std::numeric_limits<real>::epsilon();
    std::vector<std::uint64_t> seen;
    for (std::size_t row = 0; row < snapshot.row_count(); ++row) {
        const auto id = snapshot.row_id(row);
        seen.push_back(id);
        const auto source = std::find_if(expected.particles.begin(), expected.particles.end(),
                                         [&](const auto& p) { return p.id == id; });
        ASSERT_NE(source, expected.particles.end());
        const auto position = checked(snapshot.position(row));
        const auto velocity = checked(snapshot.velocity(row));
        for (int k = 0; k < 3; ++k) {
            EXPECT_LE(std::abs(static_cast<long double>(position[k]) - source->position[k]),
                      tolerance *
                          std::max(1.0L, std::abs(static_cast<long double>(source->position[k]))));
            EXPECT_LE(std::abs(static_cast<long double>(velocity[k]) - expected.velocity.at(id)[k]),
                      tolerance * std::max(1.0L, std::abs(static_cast<long double>(
                                                     expected.velocity.at(id)[k]))));
        }
        const auto pressure = checked(snapshot.pressure(row));
        const auto acceleration = checked(snapshot.pressure_acceleration(row));
        const long double density_error = std::abs(static_cast<long double>(snapshot.density(row)) -
                                                   independent_force.csr.density.at(id));
        const long double pressure_error =
            std::abs(static_cast<long double>(pressure) - independent_force.pressure.at(id));
        EXPECT_LE(density_error, tolerance * independent_force.csr.density.at(id));
        EXPECT_LE(pressure_error, tolerance * independent_force.pressure_scale.at(id));
        for (int k = 0; k < 3; ++k) {
            const long double acceleration_error =
                std::abs(static_cast<long double>(acceleration[k]) -
                         independent_force.acceleration.at(id)[k]);
            EXPECT_LE(acceleration_error,
                      4 * tolerance * independent_force.acceleration_scale.at(id)[k]);
            if (maximum) {
                maximum->position =
                    std::max(maximum->position,
                             std::abs(static_cast<long double>(position[k]) - source->position[k]));
                maximum->velocity =
                    std::max(maximum->velocity, std::abs(static_cast<long double>(velocity[k]) -
                                                         expected.velocity.at(id)[k]));
                maximum->acceleration[k] = std::max(maximum->acceleration[k], acceleration_error);
            }
        }
        if (maximum) {
            maximum->density = std::max(maximum->density, density_error);
            maximum->pressure = std::max(maximum->pressure, pressure_error);
        }
        observed[std::distance(expected.particles.begin(), source)].position = position;
    }
    ASSERT_EQ(snapshot.row_count(), expected.particles.size());
    std::sort(seen.begin(), seen.end());
    EXPECT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end());
    ASSERT_NO_FATAL_FAILURE(check_force_snapshot(snapshot, observed, eos, expected.density));
    // Neighbor identity must also match the independently evolved geometry.
    const auto csr = explicit_csr(expected.particles);
    for (std::size_t row = 0; row < snapshot.row_count(); ++row) {
        const auto index = csr.row_by_id.at(snapshot.row_id(row));
        std::vector<std::uint64_t> neighbors;
        snapshot.for_each_neighbor(row, [&](std::uint64_t id, double) { neighbors.push_back(id); });
        std::sort(neighbors.begin(), neighbors.end());
        EXPECT_EQ(neighbors,
                  (std::vector<std::uint64_t>(csr.ids.begin() + csr.offsets[index],
                                              csr.ids.begin() + csr.offsets[index + 1])));
    }
}
TEST(SphVerletWindow, ConnectedThreeDimensionalCloudChangesNeighborsAcrossSteps) {
    const sph::WcsphVerletParameters parameters{{real{.75}, real{.5}, real{3}}, .5};
    constexpr std::array input_order{5, 0, 6, 3, 7, 1, 4, 2};
    Trajectory expected;
    for (const int corner : input_order) {
        Particle particle{};
        particle.id = 100 + 17 * corner;
        particle.mass = real{1} + real{.125} * corner;
        auto& velocity = expected.velocity[particle.id];
        for (int axis = 0; axis < 3; ++axis) {
            const bool upper = (corner >> axis) & 1;
            particle.position[axis] = upper ? real{.875} : real{-.375};
            const real speed = real{.375} + real{.015625} * (corner + axis);
            velocity[axis] = upper ? -speed : speed;
        }
        expected.particles.push_back(particle);
    }
    const auto initial_csr = explicit_csr(expected.particles);
    const auto first_id = expected.particles.front().id;
    const auto first_row = initial_csr.row_by_id.at(first_id);
    ASSERT_GE(initial_csr.offsets[first_row + 1] - initial_csr.offsets[first_row], 3);
    const auto initial_force = force_oracle(expected.particles, parameters.eos);
    for (const auto acceleration : initial_force.acceleration.at(first_id))
        ASSERT_NE(acceleration, 0);

    Fields fields(expected.particles);
    const auto velocity = input_velocity(expected);
    sph::Participant3D participant(fields.get(), 1000, parameters, velocity,
                                   seed_density(expected.particles));
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_NO_FATAL_FAILURE(
        check_trajectory(checked(participant.snapshot()), expected, parameters.eos, 0));

    bool neighbors_changed = false;
    constexpr double dt = .04;
    for (std::size_t step = 1; step <= 8; ++step) {
        expected = explicit_verlet(expected, parameters.eos, dt);
        const auto csr = explicit_csr(expected.particles);
        neighbors_changed |= csr.offsets != initial_csr.offsets || csr.ids != initial_csr.ids;
        ASSERT_TRUE(simulation.advance(dt).ok());
        ASSERT_NO_FATAL_FAILURE(
            check_trajectory(checked(participant.snapshot()), expected, parameters.eos, step));
    }
    EXPECT_TRUE(neighbors_changed);
    ASSERT_TRUE(simulation.finalize().ok());
}
TEST(SphVerletWindow, UniformVelocityCrossesCellsAndStrictCfl) {
    const sph::WcsphVerletParameters parameters{{real{1}, real{.5}, real{3}}, 1.0};
    Trajectory expected{{{{8, 8, 8}, 1, 99, 0}, {{-.25, .5, .75}, 2, 11, 0}},
                        {{99, {4, 1, -2}}, {11, {4, 1, -2}}}};
    Fields fields(expected.particles);
    auto velocity = input_velocity(expected);
    sph::Participant3D participant(fields.get(), 10000, parameters, velocity,
                                   seed_density(expected.particles));
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto initial = checked(participant.snapshot());
    const double bound = participant.time_constraint(0).maximum_step;
    EXPECT_DOUBLE_EQ(bound, 1.0 / (.5 + std::sqrt(21.0)));
    EXPECT_FALSE(
        simulation.begin_window(std::nextafter(bound, std::numeric_limits<double>::infinity()))
            .ok());
    EXPECT_EQ(simulation.state(), SimulationState::prepared);
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0);
    // Exactly the accepted bound is legal; discard it before the fixed trajectory.
    ASSERT_TRUE(simulation.begin_window(bound).ok());
    ASSERT_TRUE(simulation.restore_window().ok());
    for (std::size_t step = 1; step <= 6; ++step) {
        expected = explicit_verlet(expected, parameters.eos, .125);
        ASSERT_TRUE(simulation.advance(.125).ok());
        const auto accepted = checked(participant.snapshot());
        ASSERT_NO_FATAL_FAILURE(check_trajectory(accepted, expected, parameters.eos, step));
        for (std::size_t row = 0; row < accepted.row_count(); ++row)
            EXPECT_EQ(checked(accepted.velocity(row)), (std::array<real, 3>{4, 1, -2}));
    }
    ASSERT_TRUE(simulation.finalize().ok());
    Trajectory original{{{{8, 8, 8}, 1, 99, 0}, {{-.25, .5, .75}, 2, 11, 0}},
                        {{99, {4, 1, -2}}, {11, {4, 1, -2}}}};
    ASSERT_NO_FATAL_FAILURE(check_trajectory(initial, original, parameters.eos, 0));
}
Trajectory pair_state() {
    return {{{{static_cast<real>(.05), static_cast<real>(.05), static_cast<real>(.05)}, 1, 91, 0},
             {{static_cast<real>(.8), static_cast<real>(.55), static_cast<real>(.3)}, 2, 12, 0}},
            {{91, {0, 0, 0}}, {12, {0, 0, 0}}}};
}
TEST(SphVerletWindow, UnequalMassTrajectoryMatchesIndependentCsrAndMomentum) {
    const sph::WcsphVerletParameters parameters{{real{.125}, real{.5}, real{3}}, .5};
    auto expected = pair_state();
    Fields fields(expected.particles);
    auto velocity = input_velocity(expected);
    sph::Participant3D participant(fields.get(), 1000, parameters, velocity,
                                   seed_density(expected.particles));
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto initial = checked(participant.snapshot());
    TrajectoryErrors maximum;
    const double initial_cfl = participant.time_constraint(0).maximum_step;
    for (std::size_t step = 1; step <= 12; ++step) {
        const auto next = explicit_verlet(expected, parameters.eos, .02);
        ASSERT_TRUE(simulation.advance(.02).ok());
        const auto snapshot = checked(participant.snapshot());
        ASSERT_NO_FATAL_FAILURE(check_trajectory(snapshot, next, parameters.eos, step, &maximum));
        std::array<long double, 3> momentum{}, absolute{};
        for (std::size_t row = 0; row < snapshot.row_count(); ++row) {
            const auto id = snapshot.row_id(row);
            const real mass = id == 91 ? real{1} : real{2};
            const auto v = checked(snapshot.velocity(row));
            for (int k = 0; k < 3; ++k) {
                momentum[k] += mass * static_cast<long double>(v[k]);
                absolute[k] += mass * std::abs(static_cast<long double>(v[k]));
            }
        }
        for (int k = 0; k < 3; ++k)
            EXPECT_LE(std::abs(momentum[k]),
                      64 * (step + 1) * std::numeric_limits<real>::epsilon() * absolute[k]);
        long double vmax = 0;
        for (std::size_t row = 0; row < snapshot.row_count(); ++row) {
            const auto v = checked(snapshot.velocity(row));
            long double square = 0;
            for (const real component : v)
                square += static_cast<long double>(component) * component;
            vmax = std::max(vmax, std::sqrt(square));
        }
        const double expected_cfl = parameters.cfl / (parameters.eos.sound_speed + vmax);
        EXPECT_NEAR(participant.time_constraint(simulation.time().accepted_time).maximum_step,
                    expected_cfl, 16 * std::numeric_limits<double>::epsilon() * expected_cfl);
        expected = next;
    }
    const auto latest = checked(participant.snapshot());
    EXPECT_GT(latest.stamp.layout.value(), initial.stamp.layout.value());
    EXPECT_LT(participant.time_constraint(simulation.time().accepted_time).maximum_step,
              initial_cfl);
    const auto record = [&](const char* key, long double value) {
        std::ostringstream text;
        text << std::scientific << std::setprecision(17) << value;
        ::testing::Test::RecordProperty(key, text.str());
    };
    record("trajectory_max_position_error", maximum.position);
    record("trajectory_max_velocity_error", maximum.velocity);
    record("trajectory_max_density_error", maximum.density);
    record("trajectory_max_pressure_error", maximum.pressure);
    record("trajectory_max_ax_error", maximum.acceleration[0]);
    record("trajectory_max_ay_error", maximum.acceleration[1]);
    record("trajectory_max_az_error", maximum.acceleration[2]);
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_NO_FATAL_FAILURE(check_trajectory(latest, expected, parameters.eos, 12));
}
TEST(SphVerletWindow, NoninitialRestoreAndSmallerRetryPreserveHeldSnapshots) {
    const sph::WcsphVerletParameters parameters{{real{.125}, real{.5}, real{3}}, .5};
    const auto initial = pair_state();
    Fields fields(initial.particles);
    auto velocity = input_velocity(initial);
    sph::Participant3D participant(fields.get(), 1000, parameters, velocity,
                                   seed_density(initial.particles));
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    auto accepted_reference = explicit_verlet(initial, parameters.eos, .02);
    ASSERT_TRUE(simulation.advance(.02).ok());
    const auto accepted = checked(participant.snapshot());
    const double accepted_cfl = participant.time_constraint(.02).maximum_step;
    const auto discarded_reference = explicit_verlet(accepted_reference, parameters.eos, .04);
    ASSERT_TRUE(simulation.begin_window(.04).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    const auto stamp = checked(participant.trial_stamp());
    const auto discarded = checked(participant.trial_snapshot(stamp));
    ASSERT_NO_FATAL_FAILURE(check_trajectory(discarded, discarded_reference, parameters.eos, 2));
    ASSERT_TRUE(simulation.restore_window().ok());
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, .02);
    EXPECT_FALSE(participant.trial_snapshot(stamp).ok());
    EXPECT_DOUBLE_EQ(participant.time_constraint(.02).maximum_step, accepted_cfl);
    ASSERT_NO_FATAL_FAILURE(
        check_trajectory(checked(participant.snapshot()), accepted_reference, parameters.eos, 1));
    const auto retry_reference = explicit_verlet(accepted_reference, parameters.eos, .01);
    ASSERT_TRUE(simulation.advance(.01).ok());
    const auto retried = checked(participant.snapshot());
    ASSERT_NO_FATAL_FAILURE(check_trajectory(retried, retry_reference, parameters.eos, 2));
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_NO_FATAL_FAILURE(check_trajectory(accepted, accepted_reference, parameters.eos, 1));
    ASSERT_NO_FATAL_FAILURE(check_trajectory(discarded, discarded_reference, parameters.eos, 2));
    ASSERT_NO_FATAL_FAILURE(check_trajectory(retried, retry_reference, parameters.eos, 2));
}

TEST(SphVerletWindow, GridCapacityFailureRestoresAllFieldsAndAllowsShorterStep) {
    const sph::WcsphVerletParameters parameters{{1, .25, 3}, .5};
    const Trajectory initial{{{{.125, 0, 0}, 1, 11, 0}, {{3.875, 0, 0}, 1, 22, 0}},
                             {{11, {-.5, 0, 0}}, {22, {.5, 0, 0}}}};
    Fields fields(initial.particles);
    sph::Participant3D participant(fields.get(), 2, parameters, input_velocity(initial),
                                   seed_density(initial.particles));
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto before = checked(participant.snapshot());
    const auto bytes = participant.resource_usage().controlled_bytes;
    const auto limit = participant.time_constraint(0).maximum_step;
    ASSERT_GE(limit, .5);
    // Both particles cross the outer grid boundary. Their new box spans four cells.
    const auto failed = simulation.advance(.5);
    EXPECT_EQ(failed.code(), StatusCode::resource_exhausted);
    EXPECT_EQ(simulation.state(), SimulationState::prepared);
    EXPECT_EQ(simulation.time().accepted_steps, 0u);
    EXPECT_EQ(simulation.time().accepted_time, 0);
    EXPECT_EQ(participant.time_constraint(0).maximum_step, limit);
    EXPECT_EQ(participant.resource_usage().controlled_bytes, bytes);
    EXPECT_EQ(checked(participant.snapshot()).stamp, before.stamp);
    ASSERT_NO_FATAL_FAILURE(
        check_trajectory(checked(participant.snapshot()), initial, parameters.eos, 0));
    ASSERT_TRUE(simulation.advance(.125).ok());
    ASSERT_NO_FATAL_FAILURE(check_trajectory(checked(participant.snapshot()),
                                             explicit_verlet(initial, parameters.eos, .125),
                                             parameters.eos, 1));
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_NO_FATAL_FAILURE(check_trajectory(before, initial, parameters.eos, 0));
}

TEST(SphVerletWindow, OwnsInitialBuffersDuringBallisticStep) {
    const sph::WcsphVerletParameters parameters{{1, .0625, 3}, .5};
    const Trajectory initial{{{{0, 0, 0}, 1, 11, 0}}, {{11, {.125, 0, 0}}}};
    Fields fields(initial.particles);
    auto velocities = input_velocity(initial);
    sph::Participant3D participant(fields.get(), 1, parameters, velocities,
                                   seed_density(initial.particles));
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    // Input handles belong to the caller, who may reuse them after initialization.
    velocities(0, 0) = 10;
    fields.position(0, 0) = 20;
    ASSERT_TRUE(simulation.advance(1.5).ok());
    EXPECT_EQ(simulation.time().accepted_time, 1.5);
    const auto state = checked(participant.snapshot());
    EXPECT_EQ(checked(state.velocity(0))[0], real{.125});
    EXPECT_EQ(checked(state.position(0))[0], real{.1875});
    EXPECT_EQ(state.position(1).status().code(), StatusCode::invalid_argument);
    EXPECT_EQ(state.velocity(1).status().code(), StatusCode::invalid_argument);
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(SphVerletWindow, ReusedWorkspaceIsReleasedWithoutInvalidatingRetainedSnapshot) {
    const sph::WcsphVerletParameters parameters{{1, .5, 3}, .5};
    const Trajectory initial{{{{.25, .5, .75}, 1, 11, 0}}, {{11, {0, 0, 0}}}};
    Fields fields(initial.particles);
    sph::Participant3D participant(fields.get(), 1, parameters, input_velocity(initial),
                                   seed_density(initial.particles));
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    const auto accepted_bytes = participant.resource_usage().controlled_bytes;
    ASSERT_TRUE(simulation.advance(.125).ok());
    const auto warmed_bytes = participant.resource_usage().controlled_bytes;
    EXPECT_GT(warmed_bytes, accepted_bytes);
    for (int step = 0; step < 3; ++step) {
        ASSERT_TRUE(simulation.advance(.125).ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, warmed_bytes);
    }
    {
        const auto held = checked(participant.snapshot());
        ASSERT_TRUE(simulation.finalize().ok());
        // Only the published geometry/fields remain pinned by the reader. Private
        // integration workspace must end with the participant's execution lifetime.
        EXPECT_EQ(participant.resource_usage().controlled_bytes, accepted_bytes);
        EXPECT_EQ(participant.resource_usage().snapshot_records, 1u);
        ASSERT_NO_FATAL_FAILURE(check_trajectory(held, initial, parameters.eos, 4));
    }
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    EXPECT_EQ(participant.resource_usage().snapshot_records, 0u);
}

TEST(SphVerletWindow, WorkspaceBudgetPressurePreservesHistoricalStepCapacity) {
    const sph::WcsphVerletParameters parameters{{.375, .5, 3}, .5};
    const Trajectory initial{{{{.25, .5, .75}, 1, 11, 0}, {{1.25, .5, .75}, .75, 22, 0}},
                             {{11, {.125, 0, 0}}, {22, {-.125, 0, 0}}}};
    Fields fields(initial.particles);
    const auto velocity = input_velocity(initial);
    const auto density = seed_density(initial.particles);
    const auto count = initial.particles.size();
    constexpr double dt = .125;
    std::size_t accepted_bytes{};
    {
        sph::Participant3D baseline(fields.get(), 1, parameters, velocity, density);
        ASSERT_TRUE(baseline.initialize(0).ok());
        accepted_bytes = baseline.resource_usage().controlled_bytes;
        ASSERT_TRUE(baseline.finalize().ok());
    }
    // There is room for one scratch vector. A failed step must release its
    // allocations and preserve the original publication.
    {
        const auto insufficient = accepted_bytes + count * 3 * sizeof(real);
        sph::Participant3D limited(fields.get(), 1, parameters, velocity, density, ParticipantId{1},
                                   {32, insufficient});
        Simulation simulation(limited);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.prepare().ok());
        const auto held = checked(limited.snapshot());
        const auto bound = limited.time_constraint(0).maximum_step;
        for (int attempt = 0; attempt < 2; ++attempt) {
            EXPECT_EQ(simulation.advance(dt).code(), StatusCode::resource_exhausted);
            EXPECT_EQ(simulation.state(), SimulationState::prepared);
            EXPECT_EQ(simulation.time().accepted_steps, 0u);
            EXPECT_EQ(simulation.time().accepted_time, 0);
            EXPECT_EQ(limited.time_constraint(0).maximum_step, bound);
            EXPECT_EQ(limited.resource_usage().controlled_bytes, accepted_bytes);
            EXPECT_EQ(checked(limited.snapshot()).stamp, held.stamp);
        }
        ASSERT_NO_FATAL_FAILURE(check_trajectory(held, initial, parameters.eos, 0));
        ASSERT_TRUE(simulation.finalize().ok());
    }
    // Both particles stay in one cell, so the grid index is shared across the
    // step. At pressure evaluation the original implementation needed the half
    // velocity (3), new geometry (4), corrected density (1), output fields (8),
    // and the pressure method's workspace per particle.
    const auto trial_bytes = 16 * sizeof(real) + sph::pressure_workspace_bytes_per_source(false);
    const auto limit = accepted_bytes + count * trial_bytes;
    sph::Participant3D participant(fields.get(), 1, parameters, velocity, density,
                                   ParticipantId{1}, {32, limit});
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());

    {
        const auto accepted = checked(participant.snapshot());
        ASSERT_TRUE(simulation.begin_window(dt).ok());
        ASSERT_TRUE(simulation.advance_trial().ok());
        const auto trial_stamp = checked(participant.trial_stamp());
        const auto trial = checked(participant.trial_snapshot(trial_stamp));
        ASSERT_TRUE(simulation.restore_window().ok());
        EXPECT_EQ(simulation.time().accepted_steps, 0u);
        EXPECT_EQ(checked(participant.snapshot()).stamp, accepted.stamp);
        EXPECT_FALSE(participant.trial_snapshot(trial_stamp).ok());
        const auto endpoint = explicit_verlet(initial, parameters.eos, dt);
        ASSERT_NO_FATAL_FAILURE(check_trajectory(trial, endpoint, parameters.eos, 1));
    }

    auto expected = initial;
    for (int step = 1; step <= 3; ++step) {
        expected = explicit_verlet(expected, parameters.eos, dt);
        ASSERT_TRUE(simulation.advance(dt).ok());
        EXPECT_LE(participant.resource_usage().controlled_bytes, limit);
        ASSERT_NO_FATAL_FAILURE(
            check_trajectory(checked(participant.snapshot()), expected, parameters.eos, step));
    }
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    EXPECT_EQ(participant.resource_usage().snapshot_records, 0u);
}

TEST(SphVerletWindow, RetainedTrialCanCommitWithinWorkspaceBudget) {
    const sph::WcsphVerletParameters parameters{{1, .5, 3}, .5};
    const Trajectory initial{{{{.25, .5, .75}, 1, 11, 0}}, {{11, {0, 0, 0}}}};
    Fields fields(initial.particles);
    const auto velocity = input_velocity(initial);
    const auto density = seed_density(initial.particles);
    std::size_t accepted_bytes{};
    {
        sph::Participant3D baseline(fields.get(), 1, parameters, velocity, density);
        ASSERT_TRUE(baseline.initialize(0).ok());
        accepted_bytes = baseline.resource_usage().controlled_bytes;
        ASSERT_TRUE(baseline.finalize().ok());
    }
    // Allow the old accepted state, new geometry (4 scalars), and two new field
    // banks (8 scalars each) so a reader can keep the trial across publication.
    const auto limit = accepted_bytes + 20 * sizeof(real);
    sph::Participant3D participant(fields.get(), 1, parameters, velocity, density, ParticipantId{1},
                                   {32, limit});
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_TRUE(simulation.begin_window(.125).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    {
        const auto trial = checked(participant.trial_snapshot(checked(participant.trial_stamp())));
        ASSERT_TRUE(simulation.commit_window().ok());
        EXPECT_FALSE(trial.stamp.accepted);
        ASSERT_NO_FATAL_FAILURE(check_trajectory(trial, initial, parameters.eos, 1));
        const auto accepted = checked(participant.snapshot());
        EXPECT_TRUE(accepted.stamp.accepted);
        ASSERT_NO_FATAL_FAILURE(check_trajectory(accepted, initial, parameters.eos, 1));
    }
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
    EXPECT_EQ(participant.resource_usage().snapshot_records, 0u);
}

TEST(SphVerletWindow, ChecksInitialVelocityCflAndOwnershipBeforePublishing) {
    const sph::WcsphVerletParameters parameters{{1, 1, 3}, .5};
    Fields fields({{{0, 0, 0}, 1, 1, 0}});
    sph::Velocity3D velocity("test.initial_velocity", 1);
    const auto rejects = [&](sph::WcsphVerletParameters p, sph::Velocity3D v, StatusCode reason) {
        sph::Participant3D participant(fields.get(), 1, p, v);
        Simulation simulation(participant);
        EXPECT_EQ(simulation.initialize().code(), reason);
        EXPECT_FALSE(participant.snapshot().ok());
        EXPECT_EQ(participant.resource_usage().controlled_bytes, 0u);
        EXPECT_TRUE(simulation.finalize().ok());
    };
    for (double cfl : {0., 1.001, std::numeric_limits<double>::quiet_NaN()}) {
        SCOPED_TRACE(cfl);
        rejects({parameters.eos, cfl}, velocity, StatusCode::invalid_argument);
    }
    rejects(parameters, {}, StatusCode::invalid_argument);
    velocity(0, 2) = std::numeric_limits<real>::quiet_NaN();
    rejects(parameters, velocity, StatusCode::invalid_argument);
    velocity(0, 2) = 0;
    fields.ghost(0) = 1;
    rejects(parameters, velocity, StatusCode::unsupported);
    fields.ghost(0) = 0;

    // Staged mode accepts caller geometry. Integrated mode owns each candidate.
    sph::Participant3D staged(fields.get(), 1, parameters.eos);
    ASSERT_TRUE(staged.initialize(0).ok());
    EXPECT_EQ(checked(staged.snapshot()).velocity(0).status().code(), StatusCode::unsupported);
    ASSERT_TRUE(staged.finalize().ok());
    sph::Participant3D diagnostic(fields.get(), 1, parameters, velocity);
    Simulation diagnostic_sim(diagnostic);
    ASSERT_TRUE(diagnostic_sim.initialize().ok());
    ASSERT_TRUE(diagnostic_sim.prepare().ok());
    EXPECT_GT(checked(diagnostic.snapshot()).density(0), 0);
    EXPECT_EQ(diagnostic_sim.begin_window(.125).code(), StatusCode::unsupported);
    EXPECT_EQ(diagnostic_sim.time().accepted_steps, 0u);
    EXPECT_TRUE(diagnostic_sim.finalize().ok());

    sph::Participant3D integrated(fields.get(), 1, parameters, velocity,
                                  seed_density({{{0, 0, 0}, 1, 1, 0}}));
    Simulation simulation(integrated);
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
    ASSERT_TRUE(simulation.begin_window(.125).ok());
    EXPECT_EQ(integrated.set_trial_fields(fields.get()).code(), StatusCode::unsupported);
    ASSERT_TRUE(simulation.advance_trial().ok());
    ASSERT_TRUE(simulation.commit_window().ok());
    ASSERT_TRUE(simulation.finalize().ok());

    sph::Participant3D empty(Fields({}).get(), 0, parameters, {}, sph::Values("empty.density", 0));
    Simulation empty_sim(empty);
    ASSERT_TRUE(empty_sim.initialize().ok());
    ASSERT_TRUE(empty_sim.prepare().ok());
    EXPECT_EQ(empty.time_constraint(0).maximum_step, .5);
    ASSERT_TRUE(empty_sim.advance(.5).ok());
    EXPECT_EQ(checked(empty.snapshot()).row_count(), 0u);
    ASSERT_TRUE(empty_sim.finalize().ok());

    Fields fixed({{{.25, .5, .75}, 1, 19, 1}});
    sph::Participant3D walls_only(
        Fields({}).get(), 1, parameters, {}, sph::Values("empty.density", 0),
        sph::FixedWallParticles3D{fixed.position, fixed.mass, fixed.ids});
    Simulation walls_sim(walls_only);
    ASSERT_TRUE(walls_sim.initialize().ok());
    ASSERT_TRUE(walls_sim.prepare().ok());
    const auto wall_stamp = checked(walls_only.snapshot()).stamp;
    for (int step = 0; step < 2; ++step) {
        ASSERT_TRUE(walls_sim.advance(.5).ok());
        const auto state = checked(walls_only.snapshot());
        EXPECT_EQ(state.row_count(), 0u);
        EXPECT_EQ(state.stamp.geometry, wall_stamp.geometry);
        EXPECT_EQ(state.stamp.layout, wall_stamp.layout);
        EXPECT_EQ(state.stamp.topology, wall_stamp.topology);
    }
    ASSERT_TRUE(walls_sim.finalize().ok());
    EXPECT_EQ(walls_only.resource_usage().controlled_bytes, 0u);
    EXPECT_EQ(walls_only.resource_usage().snapshot_records, 0u);
}

} // namespace
