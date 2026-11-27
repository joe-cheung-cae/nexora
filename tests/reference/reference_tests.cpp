#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include <nexora/simulation/Simulation.hpp>
#include <participants/reference/Participant.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace nexora;
using reference::Config;
using reference::Participant;
using reference::Snapshot;

static_assert(std::is_same_v<reference::Values::value_type, real>);
static_assert(std::is_same_v<decltype(Config::lambda), real>);
static_assert(std::is_same_v<decltype(Snapshot::time), double>);
static_assert(!std::is_assignable_v<decltype(std::declval<Snapshot&>().ids(0)), std::uint64_t>);
static_assert(!std::is_assignable_v<decltype(std::declval<Snapshot&>().x(0)), real>);
static_assert(!std::is_assignable_v<decltype(std::declval<Snapshot&>().q(0)), real>);
static_assert(std::is_same_v<decltype(TimeConstraint::minimum_step), double>);
static_assert(std::is_same_v<decltype(TimeConstraint::maximum_step), double>);
static_assert(std::is_same_v<decltype(TimeConstraint::fixed_quantum), std::optional<double>>);
static_assert(std::is_same_v<decltype(FieldStamp::valid_from), double>);
static_assert(std::is_same_v<decltype(FieldStamp::valid_until), double>);
static_assert(std::is_same_v<decltype(TimeState::accepted_time), double>);

void initialize(Simulation& simulation) {
    ASSERT_TRUE(simulation.initialize().ok());
}

Snapshot snapshot(const Participant& participant) {
    auto result = participant.snapshot();
    if (!result.ok()) {
        throw std::runtime_error(result.status().message());
    }
    return std::move(result).value();
}

template <class View, class Value>
void check_view(const View& view, const std::vector<Value>& expected) {
    ASSERT_EQ(view.extent(0), expected.size());
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        ASSERT_EQ(host(index), expected[index]);
    }
}

void check_descriptor(const FieldDescriptor& field, const char* name) {
    ASSERT_TRUE(field.validate().ok());
    ASSERT_EQ(field.name, name);
    ASSERT_EQ(field.association, Association::node);
    ASSERT_EQ(field.entity_set, "reference_samples");
    ASSERT_EQ(field.units, "1");
    ASSERT_EQ(field.scalar_type, real_scalar_type);
    ASSERT_EQ(field.components, 1);
    ASSERT_EQ(field.access, FieldAccess::read);
    ASSERT_EQ(field.time_level, "accepted");
}

void check_snapshot(const Snapshot& state, const std::vector<std::uint64_t>& ids,
                    const std::vector<real>& x, const std::vector<real>& q, double time,
                    std::uint64_t version) {
    ASSERT_NO_FATAL_FAILURE(check_view(state.ids, ids));
    ASSERT_NO_FATAL_FAILURE(check_view(state.x, x));
    ASSERT_NO_FATAL_FAILURE(check_view(state.q, q));
    ASSERT_EQ(state.time, time);
    ASSERT_EQ(state.version, StateVersion{version});
    ASSERT_NO_FATAL_FAILURE(check_descriptor(state.x_descriptor(), "x"));
    ASSERT_NO_FATAL_FAILURE(check_descriptor(state.q_descriptor(), "q"));
    ASSERT_NE(state.x_descriptor().id, state.q_descriptor().id);
    const auto stamp = state.stamp();
    ASSERT_TRUE(stamp.validate().ok());
    ASSERT_TRUE(stamp.accepted);
    ASSERT_EQ(stamp.accepted_version, state.version);
    ASSERT_EQ(stamp.time.physical_time, state.time);
    ASSERT_EQ(stamp.valid_from, state.time);
    ASSERT_EQ(stamp.valid_until, state.time);
}

TEST(Reference, SuccessfulStepsPublishOnceAndRetainSnapshots) {
    const Config config{{17, 3, 42}, {1.0, 2.0, -4.0}, {0.0, 1.0, 5.0}, 2.0};
    Participant participant(config);
    Simulation simulation(participant, 2.0);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto initial = snapshot(participant);
    ASSERT_TRUE(simulation.run(0.125, 1).ok());
    const auto first = snapshot(participant);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(first, config.ids, {0.75, 1.5, -3.0},
                                           {0.25, 1.5, 4.0}, 2.125, 1));
    ASSERT_TRUE(simulation.run(0.125, 1).ok());
    const auto second = snapshot(participant);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(second, config.ids, {0.5625, 1.125, -2.25},
                                           {0.4375, 1.875, 3.25}, 2.25, 2));
    ASSERT_NE(initial.x.data(), first.x.data());
    ASSERT_NE(first.x.data(), second.x.data());
    ASSERT_EQ(simulation.time().accepted_time, 2.25);
    ASSERT_EQ(simulation.time().accepted_steps, 2);
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_EQ(participant.snapshot().status().code(), StatusCode::invalid_state);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(initial, config.ids, config.x, config.q, 2.0, 0));
    ASSERT_NO_FATAL_FAILURE(check_snapshot(first, config.ids, {0.75, 1.5, -3.0},
                                           {0.25, 1.5, 4.0}, 2.125, 1));
}

TEST(Reference, LaterNumericalFailureKeepsCompletedStep) {
    const real large = std::numeric_limits<real>::max();
    const Config config{{5, 12, 19}, {real{1}, large, real{-2}},
                        {real{0}, large / real{2}, real{1}}, real{0.5}};
    Participant participant(config);
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    auto initial = snapshot(participant);
    // The middle row gives x=max/2, q=max; its next q addition overflows,
    // while both surrounding rows remain finite.
    ASSERT_EQ(simulation.run(1.0, 3).code(), StatusCode::numerical_failure);
    ASSERT_EQ(simulation.time().accepted_time, 1.0);
    ASSERT_EQ(simulation.time().accepted_steps, 1);
    auto accepted = snapshot(participant);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(accepted, config.ids, {real{0.5}, large / real{2}, real{-1}},
                                           {real{0.5}, large, real{0}}, 1.0, 1));
    ASSERT_EQ(accepted.history(0), 1);
    ASSERT_EQ(accepted.control.sequence, 1);
    ASSERT_EQ(accepted.diagnostic, 2);
    ASSERT_NO_FATAL_FAILURE(check_snapshot(initial, config.ids, config.x, config.q, 0.0, 0));
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_EQ(accepted.q(1), large);
    initial = {};
    accepted = {};
    ASSERT_EQ(participant.resource_usage().controlled_bytes, 0);
    ASSERT_EQ(participant.resource_usage().snapshot_records, 0);
}

TEST(Reference, RejectInvalidInputs) {
    const real nan = std::numeric_limits<real>::quiet_NaN();
    const real infinity = std::numeric_limits<real>::infinity();
    const double time_nan = std::numeric_limits<double>::quiet_NaN();
    const double time_infinity = std::numeric_limits<double>::infinity();
    for (const auto& config : {Config{{}, {}, {}, 1.0}, Config{{1, 1}, {1, 2}, {0, 0}, 1.0},
                               Config{{1, 2}, {1}, {0}, 1.0}, Config{{1}, {nan}, {0}, 1.0},
                               Config{{1}, {1}, {infinity}, 1.0}, Config{{1}, {1}, {0}, nan},
                               Config{{1}, {1}, {0}, infinity}, Config{{1}, {1}, {0}, 0.0},
                               Config{{1}, {1}, {0}, -1.0}}) {
        Participant participant(config);
        if (!std::isfinite(config.lambda) || config.lambda <= 0) {
            ASSERT_FALSE(participant.time_constraint().available);
            ASSERT_EQ(Participant::case_time_constraint(config.lambda).status().code(),
                      StatusCode::invalid_argument);
            ASSERT_EQ(participant.step({0.0, 0.25, 0.25}).code(), StatusCode::invalid_state);
        }
        ASSERT_EQ(participant.initialize(0.0).code(), StatusCode::invalid_argument);
        ASSERT_EQ(participant.snapshot().status().code(), StatusCode::invalid_state);
        ASSERT_TRUE(participant.finalize().ok());
    }
    Participant invalid_time;
    ASSERT_EQ(invalid_time.initialize(time_nan).code(), StatusCode::invalid_argument);
    ASSERT_TRUE(invalid_time.finalize().ok());

    Participant participant;
    const TimeInterval interval{0.0, 0.25, 0.25};
    const TimeInterval wrong_duration{0.0, 0.25, 0.5};
    ASSERT_EQ(participant.step(interval).code(), StatusCode::invalid_state);
    ASSERT_TRUE(participant.initialize(0.0).ok());
    ASSERT_EQ(participant.initialize(0.0).code(), StatusCode::invalid_state);
    for (const auto& invalid :
         {TimeInterval{-1.0, -0.75, 0.25}, wrong_duration, TimeInterval{0.0, 0.25, 0.0},
          TimeInterval{0.0, time_infinity, 0.25}, TimeInterval{0.0, 1.25, 1.25}}) {
        ASSERT_EQ(participant.step(invalid).code(), StatusCode::invalid_argument);
    }
    const auto accepted = snapshot(participant);
    ASSERT_TRUE(participant.step(interval).ok());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(accepted, {0}, {1.0}, {0.0}, 0.0, 0));
    ASSERT_NO_FATAL_FAILURE(check_snapshot(snapshot(participant), {0}, {0.75}, {0.25}, 0.25, 1));
    ASSERT_TRUE(participant.finalize().ok());
    ASSERT_EQ(participant.step(interval).code(), StatusCode::invalid_state);
}

#if !NEXORA_ENABLE_DOUBLE_PRECISION
TEST(Reference, RejectUnrepresentableRealSteps) {
    const double float_denorm = static_cast<double>(std::numeric_limits<real>::denorm_min());
    const double float_max = static_cast<double>(std::numeric_limits<real>::max());
    auto reject = [](real lambda, double dt) {
        Participant participant(Config{{0}, {1}, {0}, lambda});
        ASSERT_TRUE(participant.initialize(0.0).ok());
        const TimeInterval interval{0.0, dt, dt};
        ASSERT_EQ(participant.step(interval).code(), StatusCode::invalid_argument);
        ASSERT_NO_FATAL_FAILURE(check_snapshot(snapshot(participant), {0}, {1}, {0}, 0.0, 0));
        ASSERT_TRUE(participant.finalize().ok());
    };
    ASSERT_NO_FATAL_FAILURE(reject(real{1}, float_denorm / 4.0)); // conversion to real becomes zero
    ASSERT_NO_FATAL_FAILURE(
        reject(std::numeric_limits<real>::denorm_min(), float_max * 2.0)); // exceeds real max
    ASSERT_NO_FATAL_FAILURE(
        reject(std::numeric_limits<real>::denorm_min(), 0.25)); // real dt*lambda underflows
}
#endif

TEST(Reference, AdvertisedTimeConstraintEndpointsAreUsable) {
    const std::array lambdas{real{1}, std::numeric_limits<real>::denorm_min(),
                             std::numeric_limits<real>::max(), real{0x1.8p-1}, real{0x1.34p126},
                             real{0x1.06ee48p126}};
    for (const real lambda : lambdas) {
        SCOPED_TRACE(::testing::Message() << "lambda=" << std::hexfloat << lambda);
        Participant probe(Config{{0}, {1}, {0}, lambda});
        const TimeConstraint constraint = probe.time_constraint();
        ASSERT_TRUE(constraint.available) << Participant::case_time_constraint(lambda).status().message();
        ASSERT_TRUE(std::isfinite(constraint.minimum_step));
        ASSERT_GT(constraint.minimum_step, 0.0);
        ASSERT_TRUE(std::isfinite(constraint.maximum_step));
        ASSERT_GT(constraint.maximum_step, 0.0);
        ASSERT_LE(constraint.minimum_step, constraint.maximum_step);
        if (lambda == std::numeric_limits<real>::denorm_min()) {
            EXPECT_EQ(constraint.minimum_step, 1.0);
            EXPECT_EQ(constraint.maximum_step, static_cast<double>(std::numeric_limits<real>::max()));
        }
        ASSERT_TRUE(validate_time_constraint(constraint).ok());
        for (const double step : {constraint.minimum_step, constraint.maximum_step}) {
            ASSERT_TRUE(validate_time_step(constraint, step).ok());
            Participant participant(Config{{0}, {1}, {0}, lambda});
            Simulation simulation(participant);
            ASSERT_NO_FATAL_FAILURE(initialize(simulation));
            ASSERT_TRUE(simulation.run(step, 1).ok());
            ASSERT_EQ(simulation.time().accepted_time, step);
            const auto accepted = snapshot(participant);
            ASSERT_EQ(accepted.time, step);
            ASSERT_EQ(accepted.stamp().time.physical_time, step);
            ASSERT_EQ(accepted.stamp().valid_from, step);
            ASSERT_EQ(accepted.stamp().valid_until, step);
            ASSERT_TRUE(simulation.finalize().ok());
        }
    }
}

TEST(Reference, StableTransferHandlesBothSignsAtRealLimit) {
    constexpr real maximum = std::numeric_limits<real>::max();
    Participant participant(Config{{2, 3}, {maximum, -maximum}, {real{0}, real{0}}, real{1}});
    ASSERT_TRUE(participant.initialize(0.0).ok());
    ASSERT_TRUE(participant.step({0.0, 1.0, 1.0}).ok());
    ASSERT_NO_FATAL_FAILURE(check_snapshot(snapshot(participant), {2, 3}, {real{0}, real{0}},
                                           {maximum, -maximum}, 1.0, 1));
    ASSERT_TRUE(participant.finalize().ok());
}

TEST(Reference, RequestedDurationSurvivesEndpointRounding) {
    Participant participant;
    constexpr double initial_time = 0x1p52;
    Simulation simulation(participant, initial_time);
    ASSERT_NO_FATAL_FAILURE(initialize(simulation));
    ASSERT_TRUE(simulation.run(0.75, 1).ok());
    ASSERT_EQ(simulation.time().accepted_time, initial_time + 1.0);
    ASSERT_NO_FATAL_FAILURE(
        check_snapshot(snapshot(participant), {0}, {0.25}, {0.75}, initial_time + 1.0, 1));
    ASSERT_TRUE(simulation.finalize().ok());
}

// For this unit-amplitude, lambda=1 decay, x and q stay in [0,1]. Each
// step rounds dt*lambda, the transfer multiply, and two updates: at most
// four operations at unit scale. Eight epsilons per step conservatively
// cover these errors and their accumulation for n*epsilon << 1. The
// separate term covers converting the double time step to real.
long double real_roundoff_bound(int steps, double dt) {
    const long double eps = std::numeric_limits<real>::epsilon();
    const long double conversion =
        std::abs(static_cast<long double>(static_cast<real>(dt)) - static_cast<long double>(dt));
    return 8.0L * steps * eps + steps * conversion;
}

TEST(Reference, EulerConvergence) {
    constexpr std::array steps_by_level{16, 32, 64};
    std::array<long double, steps_by_level.size()> errors{};
    const long double exact = std::exp(-1.0L);
    std::cout << std::setprecision(17)
              << "Reference precision=" << (std::is_same_v<real, float> ? "float" : "double")
              << " epsilon=" << std::numeric_limits<real>::epsilon()
              << " long_double_digits=" << std::numeric_limits<long double>::digits << '\n';
    for (std::size_t level = 0; level < errors.size(); ++level) {
        const int steps = steps_by_level[level];
        const double dt = 1.0 / steps;
        Participant participant;
        Simulation simulation(participant);
        ASSERT_NO_FATAL_FAILURE(initialize(simulation));
        real x = 1;
        real q = 0;
        long double max_conservation_error = 0;
        const long double roundoff = real_roundoff_bound(steps, dt);
        const long double conservation_tolerance = std::is_same_v<real, double> ? 1e-12L : roundoff;
        for (int step = 0; step < steps; ++step) {
            ASSERT_TRUE(simulation.run(dt, 1).ok());
            const auto committed = snapshot(participant);
            const auto host_x =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, committed.x);
            const auto host_q =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, committed.q);
            x = host_x(0);
            q = host_q(0);
            const long double conservation_error =
                std::abs(static_cast<long double>(x) + static_cast<long double>(q) - 1.0L);
            ASSERT_LE(conservation_error, conservation_tolerance);
            max_conservation_error = std::max(max_conservation_error, conservation_error);
        }
        const auto committed = snapshot(participant);
        ASSERT_EQ(committed.time, 1.0);
        ASSERT_EQ(committed.version, StateVersion{static_cast<std::uint64_t>(steps)});
        ASSERT_EQ(simulation.time().accepted_time, 1.0);
        // Closed form is independent of the implementation's per-step loop.
        const long double euler_x = std::pow(1.0L - static_cast<long double>(dt), steps);
        const long double euler_q = 1.0L - euler_x;
        const long double x_tolerance =
            std::is_same_v<real, double> ? 1e-12L + 1e-10L * std::abs(euler_x) : roundoff;
        const long double q_tolerance =
            std::is_same_v<real, double> ? 1e-12L + 1e-10L * std::abs(euler_q) : roundoff;
        const long double x_euler_error = std::abs(static_cast<long double>(x) - euler_x);
        const long double q_euler_error = std::abs(static_cast<long double>(q) - euler_q);
        ASSERT_LE(x_euler_error, x_tolerance);
        ASSERT_LE(q_euler_error, q_tolerance);
        errors[level] = std::abs(static_cast<long double>(x) - exact);
        std::cout << "Reference Euler " << steps << " steps, x=" << x << ", q=" << q
                  << ", analytic_error=" << errors[level] << ", x_euler_error=" << x_euler_error
                  << ", q_euler_error=" << q_euler_error << ", x_tolerance=" << x_tolerance
                  << ", q_tolerance=" << q_tolerance
                  << ", max_conservation_error=" << max_conservation_error
                  << ", conservation_tolerance=" << conservation_tolerance << '\n';
        ASSERT_TRUE(simulation.finalize().ok());
    }
    for (std::size_t level = 1; level < errors.size(); ++level) {
        const long double order = std::log2(errors[level - 1] / errors[level]);
        std::cout << "Reference Euler observed order " << order << " (allowed 0.9..1.1)\n";
        ASSERT_GE(order, 0.9L);
        ASSERT_LE(order, 1.1L);
    }
}

TEST(Reference, NonDyadicStepMatchesEulerOracle) {
    // 0.1 is not exactly representable as float; compare against the requested
    // double duration, explicitly accounting for its conversion to real.
    constexpr int steps = 10;
    constexpr double dt = 0.1;
    Participant participant;
    Simulation simulation(participant);
    ASSERT_NO_FATAL_FAILURE(initialize(simulation));
    ASSERT_TRUE(simulation.run(dt, steps).ok());
    const auto result = snapshot(participant);
    const auto host_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.x);
    const auto host_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, result.q);
    const long double oracle = std::pow(1.0L - static_cast<long double>(dt), steps);
    const long double error_x = std::abs(static_cast<long double>(host_x(0)) - oracle);
    const long double error_q = std::abs(static_cast<long double>(host_q(0)) - (1.0L - oracle));
    const long double x_tolerance = std::is_same_v<real, double>
                                        ? 1e-12L + 1e-10L * std::abs(oracle)
                                        : real_roundoff_bound(steps, dt);
    const long double q_tolerance = std::is_same_v<real, double>
                                        ? 1e-12L + 1e-10L * std::abs(1.0L - oracle)
                                        : real_roundoff_bound(steps, dt);
    std::cout << std::setprecision(17) << "Reference non-dyadic dt=" << dt << ", x=" << host_x(0)
              << ", q=" << host_q(0) << ", x_euler_error=" << error_x
              << ", q_euler_error=" << error_q << ", x_tolerance=" << x_tolerance
              << ", q_tolerance=" << q_tolerance << '\n';
    ASSERT_LE(error_x, x_tolerance);
    ASSERT_LE(error_q, q_tolerance);
    ASSERT_TRUE(simulation.finalize().ok());
}

} // namespace
