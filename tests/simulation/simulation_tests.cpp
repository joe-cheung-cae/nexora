#include <nexora/simulation/Simulation.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace nexora;

void check(bool condition, const char* expression, int line) {
    if (!condition) {
        std::cerr << "line " << line << ": " << expression << '\n';
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression, __LINE__)

enum class Failure {
    none,
    initialize,
    prepare,
    begin,
    trial,
    report,
    precommit,
    commit,
    restore,
    finalize
};

// A real stateful participant: the next result depends on integration history
// and a sequence counter, so restoring only the visible value cannot pass.
class DummyParticipant final : public ParticipantIF {
  public:
    struct State {
        double value = 0.0;
        double time = 0.0;
        std::uint64_t history = 0;
        std::uint64_t sequence = 10;
        StateVersion version{};
    };

    ParticipantId id() const noexcept override {
        return ParticipantId{1};
    }
    ParticipantCapabilities capabilities() const override {
        return {rollback};
    }
    std::vector<PortDescriptor> ports() const override {
        FieldDescriptor field;
        field.id = FieldId{1};
        field.name = "dummy_value";
        field.units = invalid_port ? "" : "1";
        const PortDescriptor port{PortId{1}, SupportId{1}, PortDirection::output, field};
        return duplicate_port ? std::vector<PortDescriptor>{port, port}
                              : std::vector<PortDescriptor>{port};
    }
    TimeConstraint time_constraint(double accepted_time) const override {
        last_constraint_time = accepted_time;
        return constraint;
    }
    Status initialize(double initial_time) override {
        calls.push_back("initialize");
        state.time = initial_time;
        return status_for(Failure::initialize);
    }
    Status prepare() override {
        calls.push_back("prepare");
        return status_for(Failure::prepare);
    }
    Status begin_window(const TimeInterval&) override {
        calls.push_back("begin");
        auto status = status_for(Failure::begin);
        if (status.ok()) {
            checkpoint = state;
        }
        return status;
    }
    Result<TrialReport> advance_trial(const TimeInterval& interval) override {
        calls.push_back("trial");
        CHECK(checkpoint.has_value());
        state.value += interval.duration() * static_cast<double>(state.history + 1);
        ++state.history;
        ++state.sequence;
        state.time = interval.end;
        auto status = status_for(Failure::trial);
        if (!status.ok()) {
            return status;
        }
        if (failure == Failure::report) {
            failure = Failure::none;
            return TrialReport{interval.begin, true};
        }
        return TrialReport{interval.end, true};
    }
    Status prepare_commit(const TimeInterval&) override {
        calls.push_back("precommit");
        return status_for(Failure::precommit);
    }
    Status commit_window(const TimeInterval&) override {
        calls.push_back("commit");
        state.version = StateVersion{state.version.value() + 1};
        auto status = status_for(Failure::commit);
        if (status.ok()) {
            ++published_outputs;
            checkpoint.reset();
        }
        return status;
    }
    Status restore_window() override {
        calls.push_back("restore");
        if (fail_restore) {
            return {StatusCode::execution_failure, "injected rollback failure"};
        }
        auto status = status_for(Failure::restore);
        if (status.ok()) {
            CHECK(checkpoint.has_value());
            state = *checkpoint;
            checkpoint.reset();
        }
        return status;
    }
    Status finalize() override {
        calls.push_back("finalize");
        auto status = status_for(Failure::finalize);
        if (status.ok()) {
            checkpoint.reset();
        }
        return status;
    }

    Failure failure = Failure::none;
    State state;
    std::optional<State> checkpoint;
    TimeConstraint constraint{1.0, 0.0, std::nullopt};
    bool rollback = true;
    bool invalid_port = false;
    bool duplicate_port = false;
    bool fail_restore = false;
    int published_outputs = 0;
    mutable double last_constraint_time = -1.0;
    std::vector<std::string> calls;

  private:
    Status status_for(Failure point) {
        if (failure == point) {
            failure = Failure::none;
            return {StatusCode::execution_failure, "injected participant failure"};
        }
        return {};
    }
};

void prepare(Simulation& simulation) {
    CHECK(simulation.initialize().ok());
    CHECK(simulation.prepare().ok());
}

void lifecycle_and_replay() {
    DummyParticipant participant;
    Simulation simulation(participant);
    CHECK(!simulation.advance(0.5).ok());
    CHECK(!simulation.prepare().ok());
    CHECK(participant.calls.empty());
    prepare(simulation);
    CHECK(!simulation.initialize().ok());
    CHECK(!simulation.commit_window().ok());
    CHECK(!simulation.restore_window().ok());
    CHECK(simulation.begin_window(0.5).ok());
    CHECK(!simulation.begin_window(0.5).ok());
    CHECK(simulation.advance_trial().ok());
    CHECK(!simulation.advance_trial().ok());
    CHECK(participant.state.value == 0.5);
    CHECK(participant.state.history == 1);
    CHECK(participant.state.sequence == 11);
    CHECK(participant.state.version == StateVersion{0});
    CHECK(simulation.time().accepted_time == 0.0);
    CHECK(participant.published_outputs == 0);
    CHECK(simulation.restore_window().ok());
    CHECK(participant.state.value == 0.0);
    CHECK(participant.state.history == 0);
    CHECK(participant.state.sequence == 10);
    CHECK(simulation.advance(0.25).ok());
    CHECK(simulation.advance(0.25).ok());
    CHECK(simulation.time().accepted_time == 0.5);
    CHECK(simulation.time().accepted_steps == 2);
    CHECK(simulation.time().trial_attempts == 3);
    CHECK(participant.state.value == 0.75);
    CHECK(participant.state.history == 2);
    CHECK(participant.state.sequence == 12);
    CHECK(participant.state.version == StateVersion{2});
    CHECK(participant.published_outputs == 2);
    CHECK(participant.last_constraint_time == 0.25);
    CHECK(simulation.finalize().ok());
    CHECK(simulation.state() == SimulationState::finalized);
    CHECK(!simulation.advance(0.1).ok());
    CHECK(!simulation.finalize().ok());

    DummyParticipant fresh;
    Simulation reference(fresh);
    prepare(reference);
    CHECK(reference.advance(0.25).ok());
    CHECK(reference.advance(0.25).ok());
    CHECK(participant.state.value == fresh.state.value);
    CHECK(participant.state.history == fresh.state.history);
    CHECK(participant.state.sequence == fresh.state.sequence);
    CHECK(reference.finalize().ok());
}

void failures_and_retry() {
    for (const auto failure :
         {Failure::begin, Failure::trial, Failure::report, Failure::precommit, Failure::commit}) {
        DummyParticipant participant;
        Simulation simulation(participant, 2.0);
        prepare(simulation);
        CHECK(simulation.advance(0.25).ok());
        const auto accepted = participant.state;
        participant.failure = failure;
        CHECK(!simulation.advance(0.5).ok());
        CHECK(simulation.state() == SimulationState::prepared);
        CHECK(simulation.time().accepted_time == 2.25);
        CHECK(simulation.time().accepted_steps == 1);
        CHECK(participant.state.time == accepted.time);
        CHECK(participant.state.value == accepted.value);
        CHECK(participant.state.history == accepted.history);
        CHECK(participant.state.sequence == accepted.sequence);
        CHECK(participant.state.version == accepted.version);
        CHECK(participant.published_outputs == 1);
        CHECK(!participant.checkpoint.has_value());
        CHECK(simulation.advance(0.25).ok());
        CHECK(simulation.time().accepted_time == 2.5);
        CHECK(participant.state.value == 0.75);
        CHECK(simulation.finalize().ok());
    }
}

void failed_restore_and_finalization() {
    DummyParticipant participant;
    Simulation simulation(participant);
    prepare(simulation);
    CHECK(simulation.begin_window(0.5).ok());
    CHECK(simulation.advance_trial().ok());
    participant.failure = Failure::restore;
    CHECK(!simulation.restore_window().ok());
    CHECK(simulation.state() == SimulationState::failed);
    CHECK(simulation.time().accepted_time == 0.0);
    CHECK(simulation.time().accepted_steps == 0);
    CHECK(!simulation.advance(0.1).ok());
    CHECK(!simulation.commit_window().ok());
    CHECK(simulation.finalize().ok());
    CHECK(!participant.checkpoint.has_value());

    DummyParticipant double_failure;
    Simulation poisoned(double_failure);
    prepare(poisoned);
    double_failure.failure = Failure::trial;
    double_failure.fail_restore = true;
    const auto status = poisoned.advance(0.5);
    CHECK(!status.ok());
    CHECK(status.message().find("rollback failed") != std::string::npos);
    CHECK(poisoned.state() == SimulationState::failed);
    CHECK(poisoned.time().accepted_time == 0.0);
    CHECK(!poisoned.advance(0.1).ok());
    CHECK(poisoned.finalize().ok());

    DummyParticipant abandoned;
    Simulation cleanup(abandoned);
    prepare(cleanup);
    CHECK(cleanup.begin_window(0.5).ok());
    CHECK(cleanup.advance_trial().ok());
    CHECK(cleanup.finalize().ok());
    CHECK(abandoned.state.value == 0.0);
    CHECK(abandoned.published_outputs == 0);
    CHECK(abandoned.calls[abandoned.calls.size() - 2] == "restore");
    CHECK(abandoned.calls.back() == "finalize");

    DummyParticipant cleanup_failure;
    Simulation retry_cleanup(cleanup_failure);
    prepare(retry_cleanup);
    cleanup_failure.failure = Failure::finalize;
    CHECK(!retry_cleanup.finalize().ok());
    CHECK(retry_cleanup.state() == SimulationState::failed);
    CHECK(retry_cleanup.finalize().ok());

    for (auto failure : {Failure::initialize, Failure::prepare}) {
        DummyParticipant broken;
        Simulation failed(broken);
        broken.failure = failure;
        if (failure == Failure::prepare) {
            CHECK(failed.initialize().ok());
            CHECK(!failed.prepare().ok());
        } else {
            CHECK(!failed.initialize().ok());
        }
        CHECK(failed.state() == SimulationState::failed);
        CHECK(!failed.advance(0.1).ok());
        CHECK(failed.finalize().ok());
    }
}

void constraints_and_metadata() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (double step : {0.0, -1.0, nan, infinity, 1.5}) {
        DummyParticipant participant;
        Simulation simulation(participant);
        prepare(simulation);
        CHECK(!simulation.advance(step).ok());
        CHECK(participant.calls.size() == 2);
        CHECK(simulation.time().trial_attempts == 0);
        CHECK(simulation.finalize().ok());
    }
    CHECK(!validate_time_constraint({0.0, 0.0, {}}).ok());
    CHECK(!validate_time_constraint({1.0, 2.0, {}}).ok());
    CHECK(!validate_time_constraint({1.0, 0.0, 0.0}).ok());
    CHECK(!validate_time_constraint({1.0, nan, {}}).ok());
    CHECK(!validate_time_constraint({1.0, 0.0, infinity}).ok());
    CHECK(!validate_time_step({1.0, 0.2, {}}, 0.1).ok());
    CHECK(!validate_time_step({1.0, 0.0, 0.1}, 0.15).ok());
    CHECK(validate_time_step({1.0, 0.0, 0.1}, 0.3).ok());
    CHECK(validate_time_step({1e15 + 1.0, 0.0, 1.0}, 1e15).ok());
    CHECK(!validate_time_step({1e15 + 1.0, 0.0, 1.0}, 1e15 + 0.5).ok());
    CHECK(!validate_time_step({1e15 + 1.0, 0.0, 1.0}, 1e15 + 0.25).ok());

    DummyParticipant quantized;
    quantized.constraint = {1.0, 0.1, 0.1};
    Simulation simulation(quantized);
    prepare(simulation);
    CHECK(!simulation.advance(0.15).ok());
    CHECK(simulation.advance(0.3).ok());
    CHECK(simulation.finalize().ok());

    DummyParticipant decimal;
    decimal.constraint = {0.1, 0.1, 0.1};
    Simulation repeating(decimal);
    prepare(repeating);
    for (int index = 0; index < 1000; ++index) {
        CHECK(repeating.advance(0.1).ok());
    }
    CHECK(repeating.time().accepted_steps == 1000);
    CHECK(std::abs(repeating.time().accepted_time - 100.0) < 1e-10);
    CHECK(repeating.finalize().ok());

    for (int invalid = 0; invalid < 3; ++invalid) {
        DummyParticipant participant;
        participant.rollback = invalid != 0;
        participant.invalid_port = invalid == 1;
        participant.duplicate_port = invalid == 2;
        Simulation rejected(participant);
        CHECK(!rejected.initialize().ok());
        CHECK(participant.calls.empty());
    }
    DummyParticipant nonfinite;
    Simulation bad_time(nonfinite, nan);
    CHECK(!bad_time.initialize().ok());
    CHECK(nonfinite.calls.empty());

    DummyParticipant imprecise;
    Simulation too_small(imprecise, 0x1p54);
    prepare(too_small);
    CHECK(!too_small.advance(1.0).ok());
    CHECK(imprecise.calls.size() == 2);
    CHECK(too_small.finalize().ok());

    DummyParticipant rounded;
    rounded.constraint = {0.75, 0.75, 0.75};
    Simulation rounded_time(rounded, 0x1p52);
    prepare(rounded_time);
    CHECK(rounded_time.advance(0.75).ok());
    CHECK(rounded_time.time().accepted_time == 0x1p52 + 1.0);
    CHECK(rounded.state.value == 0.75);
    CHECK(rounded_time.finalize().ok());
}

} // namespace

int main() {
    lifecycle_and_replay();
    failures_and_retry();
    failed_restore_and_finalization();
    constraints_and_metadata();
}
