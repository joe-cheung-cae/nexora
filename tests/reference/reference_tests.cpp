#include <Kokkos_Core.hpp>
#include <nexora/simulation/Simulation.hpp>
#include <participants/reference/Participant.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
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

static_assert(!std::is_assignable_v<decltype(std::declval<Snapshot&>().ids(0)), std::uint64_t>);
static_assert(!std::is_assignable_v<decltype(std::declval<Snapshot&>().x(0)), double>);
static_assert(!std::is_assignable_v<decltype(std::declval<Snapshot&>().q(0)), double>);

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        throw std::runtime_error("line " + std::to_string(line) + " failed check " + expression);
    }
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

void reject_uninitialized_runtime() {
    CHECK(!Kokkos::is_initialized());
    static std::size_t allocations = 0;
    Kokkos::Tools::Experimental::set_allocate_data_callback(
        [](Kokkos::Tools::SpaceHandle, const char*, const void*, std::uint64_t) { ++allocations; });
    Participant participant;
    const auto status = participant.initialize(0.0);
    Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);

    CHECK(status.code() == StatusCode::invalid_state);
    CHECK(allocations == 0);
    CHECK(!Kokkos::is_initialized());
    CHECK(participant.snapshot().status().code() == StatusCode::invalid_state);
    CHECK(participant.finalize().ok());
    CHECK(participant.finalize().code() == StatusCode::invalid_state);
}

void prepare(Simulation& simulation) {
    CHECK(simulation.initialize().ok());
    CHECK(simulation.prepare().ok());
}

Snapshot snapshot(const Participant& participant) {
    auto result = participant.snapshot();
    CHECK(result.ok());
    return std::move(result).value();
}

template <class View, class Value>
void check_view(const View& view, const std::vector<Value>& expected) {
    CHECK(view.extent(0) == expected.size());
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        CHECK(host(index) == expected[index]);
    }
}

void check_descriptor(const FieldDescriptor& field, const char* name, const Snapshot& state) {
    CHECK(field.validate().ok());
    CHECK(field.name == name);
    CHECK(field.association == Association::node);
    CHECK(field.entity_set == "reference_samples");
    CHECK(field.units == "1");
    CHECK(field.scalar_type == ScalarType::float64);
    CHECK(field.components == 1);
    CHECK(field.access == FieldAccess::read);
    CHECK(field.state_version == state.version);
    CHECK(field.valid_from == state.time && field.valid_until == state.time);
}

void check_snapshot(const Snapshot& state, const std::vector<std::uint64_t>& ids,
                    const std::vector<double>& x, const std::vector<double>& q, double time,
                    std::uint64_t version, std::uint64_t attempt) {
    check_view(state.ids, ids);
    check_view(state.x, x);
    check_view(state.q, q);
    CHECK(state.time == time);
    CHECK(state.version == StateVersion{version});
    CHECK(state.attempt == attempt);
    check_descriptor(state.x_descriptor(), "x", state);
    check_descriptor(state.q_descriptor(), "q", state);
    CHECK(state.x_descriptor().id != state.q_descriptor().id);
}

void check_same_accepted_storage(const Snapshot& before, const Snapshot& after) {
    CHECK(before.ids.data() == after.ids.data());
    CHECK(before.x.data() == after.x.data());
    CHECK(before.q.data() == after.q.data());
}

void lifecycle_and_replay() {
    const Config config{{17, 3, 42}, {1.0, 2.0, -4.0}, {0.0, 1.0, 5.0}, 2.0};
    constexpr double step = 0.125;
    constexpr double start = 2.0;
    constexpr double first_time = start + step;
    constexpr double second_time = first_time + step;
    const std::vector<double> first_x{0.75, 1.5, -3.0}, first_q{0.25, 1.5, 4.0};
    const std::vector<double> second_x{0.5625, 1.125, -2.25}, second_q{0.4375, 1.875, 3.25};
    std::optional<Snapshot> initial, first, second;
    {
        Participant participant(config);
        Simulation simulation(participant, start);
        prepare(simulation);
        initial.emplace(snapshot(participant));
        check_snapshot(*initial, config.ids, config.x, config.q, start, 0, 0);
        CHECK(simulation.begin_window(0.75).code() == StatusCode::invalid_argument);

        CHECK(simulation.begin_window(step).ok());
        CHECK(simulation.advance_trial().ok());
        CHECK(simulation.state() == SimulationState::trial_advanced);
        CHECK(simulation.time().accepted_time == start);
        CHECK(simulation.time().accepted_steps == 0);
        CHECK(simulation.time().trial_attempts == 1);
        const auto during_trial = snapshot(participant);
        check_same_accepted_storage(*initial, during_trial);
        check_snapshot(during_trial, config.ids, config.x, config.q, start, 0, 0);
        CHECK(simulation.restore_window().ok());
        CHECK(simulation.state() == SimulationState::prepared);
        const auto restored = snapshot(participant);
        check_same_accepted_storage(*initial, restored);
        check_snapshot(restored, config.ids, config.x, config.q, start, 0, 0);

        CHECK(simulation.advance(step).ok());
        first.emplace(snapshot(participant));
        check_snapshot(*first, config.ids, first_x, first_q, first_time, 1, 2);
        CHECK(initial->x.data() != first->x.data() && initial->q.data() != first->q.data());
        CHECK(simulation.advance(step).ok());
        second.emplace(snapshot(participant));
        check_snapshot(*second, config.ids, second_x, second_q, second_time, 2, 3);
        CHECK(first->x.data() != second->x.data() && first->q.data() != second->q.data());
        CHECK(simulation.time().accepted_time == second_time);
        CHECK(simulation.time().accepted_steps == 2);
        CHECK(simulation.time().trial_attempts == 3);

        CHECK(simulation.begin_window(step).ok());
        CHECK(simulation.advance_trial().ok());
        check_same_accepted_storage(*second, snapshot(participant));
        CHECK(simulation.finalize().ok());
        CHECK(simulation.state() == SimulationState::finalized);
        CHECK(simulation.time().accepted_time == second_time);
        CHECK(simulation.time().accepted_steps == 2);
        CHECK(simulation.time().trial_attempts == 4);
        CHECK(participant.snapshot().status().code() == StatusCode::invalid_state);
        check_snapshot(*second, config.ids, second_x, second_q, second_time, 2, 3);
    }
    check_snapshot(*initial, config.ids, config.x, config.q, start, 0, 0);
    check_snapshot(*first, config.ids, first_x, first_q, first_time, 1, 2);
    check_snapshot(*second, config.ids, second_x, second_q, second_time, 2, 3);

    Participant fresh(config);
    Simulation replay(fresh, start);
    prepare(replay);
    CHECK(replay.advance(step).ok());
    check_snapshot(snapshot(fresh), config.ids, first_x, first_q, first_time, 1, 1);
    CHECK(replay.advance(step).ok());
    check_snapshot(snapshot(fresh), config.ids, second_x, second_q, second_time, 2, 2);
    CHECK(replay.finalize().ok());
}

void numerical_failure_and_retry() {
    const double large = std::numeric_limits<double>::max();
    const Config config{{12}, {large}, {large / 2.0}, 1.0};
    Participant participant(config);
    Simulation simulation(participant);
    prepare(simulation);
    const auto initial = snapshot(participant);

    CHECK(simulation.advance(1.0).code() == StatusCode::numerical_failure);
    CHECK(simulation.state() == SimulationState::prepared);
    CHECK(simulation.time().accepted_time == 0.0);
    CHECK(simulation.time().accepted_steps == 0);
    CHECK(simulation.time().trial_attempts == 1);
    const auto restored = snapshot(participant);
    check_same_accepted_storage(initial, restored);
    check_snapshot(restored, config.ids, config.x, config.q, 0.0, 0, 0);

    CHECK(simulation.advance(0.25).ok());
    CHECK(simulation.time().accepted_time == 0.25);
    CHECK(simulation.time().accepted_steps == 1);
    CHECK(simulation.time().trial_attempts == 2);
    const double transfer = large * 0.25;
    const std::vector<double> expected_x{large - transfer};
    const std::vector<double> expected_q{large / 2.0 + transfer};
    const auto retried = snapshot(participant);
    check_snapshot(retried, config.ids, expected_x, expected_q, 0.25, 1, 2);
    check_snapshot(initial, config.ids, config.x, config.q, 0.0, 0, 0);

    Participant fresh(config);
    Simulation direct(fresh);
    prepare(direct);
    CHECK(direct.advance(0.25).ok());
    CHECK(direct.time().accepted_time == simulation.time().accepted_time);
    CHECK(direct.time().accepted_steps == simulation.time().accepted_steps);
    CHECK(direct.time().trial_attempts == 1);
    check_snapshot(snapshot(fresh), config.ids, expected_x, expected_q, retried.time,
                   retried.version.value(), 1);
    CHECK(direct.finalize().ok());
    CHECK(simulation.finalize().ok());
}

void reject_invalid_inputs() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (const auto& config : {Config{{}, {}, {}, 1.0}, Config{{1, 1}, {1, 2}, {0, 0}, 1.0},
                               Config{{1, 2}, {1}, {0}, 1.0}, Config{{1}, {nan}, {0}, 1.0},
                               Config{{1}, {1}, {infinity}, 1.0}, Config{{1}, {1}, {0}, nan},
                               Config{{1}, {1}, {0}, infinity}, Config{{1}, {1}, {0}, 0.0},
                               Config{{1}, {1}, {0}, -1.0}}) {
        Participant participant(config);
        CHECK(participant.initialize(0.0).code() == StatusCode::invalid_argument);
        CHECK(participant.snapshot().status().code() == StatusCode::invalid_state);
        CHECK(participant.finalize().ok());
    }
    Participant invalid_time;
    CHECK(invalid_time.initialize(nan).code() == StatusCode::invalid_argument);
    CHECK(invalid_time.finalize().ok());

    Participant participant;
    const TimeInterval interval{0.0, 0.25, 0.25};
    const TimeInterval wrong_duration{0.0, 0.25, 0.5};
    CHECK(participant.prepare().code() == StatusCode::invalid_state);
    CHECK(participant.begin_window(interval).code() == StatusCode::invalid_state);
    CHECK(participant.initialize(0.0).ok());
    CHECK(participant.initialize(0.0).code() == StatusCode::invalid_state);
    CHECK(participant.prepare().ok());
    CHECK(participant.advance_trial(interval).status().code() == StatusCode::invalid_state);
    for (const auto& invalid : {TimeInterval{-1.0, -0.75, 0.25}, wrong_duration,
                                TimeInterval{0.0, 0.25, 0.0}, TimeInterval{0.0, infinity, 0.25},
                                TimeInterval{0.0, 1.25, 1.25}}) {
        CHECK(participant.begin_window(invalid).code() == StatusCode::invalid_argument);
    }
    CHECK(participant.restore_window().code() == StatusCode::invalid_state);
    const auto accepted = snapshot(participant);
    CHECK(participant.begin_window(interval).ok());
    CHECK(participant.begin_window(interval).code() == StatusCode::invalid_state);
    CHECK(participant.advance_trial(wrong_duration).status().code() == StatusCode::invalid_argument);
    const auto trial = participant.advance_trial(interval);
    CHECK(trial.ok());
    CHECK(trial.value().reached_time == interval.end);
    CHECK(trial.value().physical_state_valid);
    CHECK(participant.commit_window(interval).code() == StatusCode::invalid_state);
    CHECK(participant.prepare_commit(wrong_duration).code() == StatusCode::invalid_argument);
    CHECK(participant.prepare_commit(interval).ok());
    const auto before_commit = snapshot(participant);
    check_same_accepted_storage(accepted, before_commit);
    check_snapshot(before_commit, {0}, {1.0}, {0.0}, 0.0, 0, 0);
    CHECK(participant.commit_window(wrong_duration).code() == StatusCode::invalid_argument);
    CHECK(participant.restore_window().ok());
    CHECK(participant.restore_window().code() == StatusCode::invalid_state);
    check_same_accepted_storage(accepted, snapshot(participant));
    CHECK(participant.finalize().ok());
}

void requested_duration_survives_endpoint_rounding() {
    Participant participant;
    constexpr double initial_time = 0x1p52;
    Simulation simulation(participant, initial_time);
    prepare(simulation);
    CHECK(simulation.advance(0.75).ok());
    CHECK(simulation.time().accepted_time == initial_time + 1.0);
    check_snapshot(snapshot(participant), {0}, {0.25}, {0.75}, initial_time + 1.0, 1, 1);
    CHECK(simulation.finalize().ok());
}

void euler_convergence() {
    constexpr std::array steps_by_level{16, 32, 64};
    std::array<double, steps_by_level.size()> errors{};
    const double exact = std::exp(-1.0);
    std::cout << std::setprecision(17);
    for (std::size_t level = 0; level < errors.size(); ++level) {
        const int steps = steps_by_level[level];
        const double dt = 1.0 / steps;
        Participant participant;
        Simulation simulation(participant);
        prepare(simulation);
        double x = 1.0;
        double q = 0.0;
        double max_conservation_error = 0.0;
        for (int step = 0; step < steps; ++step) {
            CHECK(simulation.advance(dt).ok());
            const auto committed = snapshot(participant);
            const auto host_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, committed.x);
            const auto host_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, committed.q);
            x = host_x(0);
            q = host_q(0);
            const double conservation_error = std::abs(x + q - 1.0);
            CHECK(conservation_error <= 1e-12);
            max_conservation_error = std::max(max_conservation_error, conservation_error);
        }
        const auto committed = snapshot(participant);
        CHECK(committed.time == 1.0);
        CHECK(committed.version == StateVersion{static_cast<std::uint64_t>(steps)});
        CHECK(simulation.time().accepted_time == 1.0);
        const double euler_x = std::pow(1.0 - dt, steps);
        const double euler_q = 1.0 - euler_x;
        CHECK(std::abs(x - euler_x) <= 1e-12 + 1e-10 * std::abs(euler_x));
        CHECK(std::abs(q - euler_q) <= 1e-12 + 1e-10 * std::abs(euler_q));
        errors[level] = std::abs(x - exact);
        std::cout << "Reference Euler " << steps << " steps, x=" << x << ", q=" << q
                  << ", x_error=" << errors[level] << ", max_conservation_error="
                  << max_conservation_error << '\n';
        CHECK(simulation.finalize().ok());
    }
    for (std::size_t level = 1; level < errors.size(); ++level) {
        const double order = std::log2(errors[level - 1] / errors[level]);
        CHECK(order >= 0.9 && order <= 1.1);
        std::cout << "Reference Euler observed order " << order << '\n';
    }
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        reject_uninitialized_runtime();
        Kokkos::ScopeGuard runtime(argc, argv);
        lifecycle_and_replay();
        numerical_failure_and_retry();
        reject_invalid_inputs();
        requested_duration_survives_endpoint_rounding();
        euler_convergence();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
