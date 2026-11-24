#include <nexora/simulation/Simulation.hpp>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace nexora;

enum class Failure {
    none,
    initialize,
    prepare,
    begin,
    trial,
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
        real value = real{0};
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
    Status advance_trial(const TimeInterval& interval) override {
        calls.push_back("trial");
        if (!checkpoint) {
            ADD_FAILURE() << "advance_trial called without a checkpoint";
            return Status{StatusCode::invalid_state, "missing checkpoint"};
        }
        state.value += static_cast<real>(interval.duration()) * static_cast<real>(state.history + 1);
        ++state.history;
        ++state.sequence;
        state.time = interval.end;
        return status_for(Failure::trial);
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
            if (!checkpoint) {
                ADD_FAILURE() << "restore_window called without a checkpoint";
                return {StatusCode::invalid_state, "missing checkpoint"};
            }
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
    ASSERT_TRUE(simulation.initialize().ok());
    ASSERT_TRUE(simulation.prepare().ok());
}

TEST(SimulationTest, LifecycleAndReplay) {
    DummyParticipant participant;
    Simulation simulation(participant);
    ASSERT_FALSE(simulation.advance(0.5).ok());
    ASSERT_FALSE(simulation.prepare().ok());
    ASSERT_TRUE(participant.calls.empty());
    ASSERT_NO_FATAL_FAILURE(prepare(simulation));
    ASSERT_FALSE(simulation.initialize().ok());
    ASSERT_FALSE(simulation.commit_window().ok());
    ASSERT_FALSE(simulation.restore_window().ok());
    ASSERT_TRUE(simulation.begin_window(0.5).ok());
    ASSERT_FALSE(simulation.begin_window(0.5).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    ASSERT_FALSE(simulation.advance_trial().ok());
    ASSERT_EQ(participant.state.value, 0.5);
    ASSERT_EQ(participant.state.history, 1);
    ASSERT_EQ(participant.state.sequence, 11);
    ASSERT_EQ(participant.state.version, StateVersion{0});
    ASSERT_EQ(simulation.time().accepted_time, 0.0);
    ASSERT_EQ(participant.published_outputs, 0);
    ASSERT_TRUE(simulation.restore_window().ok());
    ASSERT_EQ(participant.state.value, 0.0);
    ASSERT_EQ(participant.state.history, 0);
    ASSERT_EQ(participant.state.sequence, 10);
    ASSERT_TRUE(simulation.advance(0.25).ok());
    ASSERT_TRUE(simulation.advance(0.25).ok());
    ASSERT_EQ(simulation.time().accepted_time, 0.5);
    ASSERT_EQ(simulation.time().accepted_steps, 2);
    ASSERT_EQ(simulation.time().trial_attempts, 3);
    ASSERT_EQ(participant.state.value, 0.75);
    ASSERT_EQ(participant.state.history, 2);
    ASSERT_EQ(participant.state.sequence, 12);
    ASSERT_EQ(participant.state.version, StateVersion{2});
    ASSERT_EQ(participant.published_outputs, 2);
    ASSERT_EQ(participant.last_constraint_time, 0.25);
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_EQ(simulation.state(), SimulationState::finalized);
    ASSERT_FALSE(simulation.advance(0.1).ok());
    ASSERT_FALSE(simulation.finalize().ok());

    DummyParticipant fresh;
    Simulation reference(fresh);
    ASSERT_NO_FATAL_FAILURE(prepare(reference));
    ASSERT_TRUE(reference.advance(0.25).ok());
    ASSERT_TRUE(reference.advance(0.25).ok());
    ASSERT_EQ(participant.state.value, fresh.state.value);
    ASSERT_EQ(participant.state.history, fresh.state.history);
    ASSERT_EQ(participant.state.sequence, fresh.state.sequence);
    ASSERT_TRUE(reference.finalize().ok());
}

TEST(SimulationTest, FailuresAndRetry) {
    for (const auto failure :
         {Failure::begin, Failure::trial, Failure::precommit, Failure::commit}) {
        DummyParticipant participant;
        Simulation simulation(participant, 2.0);
        ASSERT_NO_FATAL_FAILURE(prepare(simulation));
        ASSERT_TRUE(simulation.advance(0.25).ok());
        const auto accepted = participant.state;
        participant.failure = failure;
        ASSERT_FALSE(simulation.advance(0.5).ok());
        ASSERT_EQ(simulation.state(), SimulationState::prepared);
        ASSERT_EQ(simulation.time().accepted_time, 2.25);
        ASSERT_EQ(simulation.time().accepted_steps, 1);
        ASSERT_EQ(participant.state.time, accepted.time);
        ASSERT_EQ(participant.state.value, accepted.value);
        ASSERT_EQ(participant.state.history, accepted.history);
        ASSERT_EQ(participant.state.sequence, accepted.sequence);
        ASSERT_EQ(participant.state.version, accepted.version);
        ASSERT_EQ(participant.published_outputs, 1);
        ASSERT_FALSE(participant.checkpoint.has_value());
        ASSERT_TRUE(simulation.advance(0.25).ok());
        ASSERT_EQ(simulation.time().accepted_time, 2.5);
        ASSERT_EQ(participant.state.value, 0.75);
        ASSERT_TRUE(simulation.finalize().ok());
    }
}

TEST(SimulationTest, FailedRestoreAndFinalization) {
    DummyParticipant participant;
    Simulation simulation(participant);
    ASSERT_NO_FATAL_FAILURE(prepare(simulation));
    ASSERT_TRUE(simulation.begin_window(0.5).ok());
    ASSERT_TRUE(simulation.advance_trial().ok());
    participant.failure = Failure::restore;
    ASSERT_FALSE(simulation.restore_window().ok());
    ASSERT_EQ(simulation.state(), SimulationState::failed);
    ASSERT_EQ(simulation.time().accepted_time, 0.0);
    ASSERT_EQ(simulation.time().accepted_steps, 0);
    ASSERT_FALSE(simulation.advance(0.1).ok());
    ASSERT_FALSE(simulation.commit_window().ok());
    ASSERT_TRUE(simulation.finalize().ok());
    ASSERT_FALSE(participant.checkpoint.has_value());

    DummyParticipant double_failure;
    Simulation poisoned(double_failure);
    ASSERT_NO_FATAL_FAILURE(prepare(poisoned));
    double_failure.failure = Failure::trial;
    double_failure.fail_restore = true;
    const auto status = poisoned.advance(0.5);
    ASSERT_FALSE(status.ok());
    ASSERT_TRUE(status.message().find("rollback failed") != std::string::npos);
    ASSERT_EQ(poisoned.state(), SimulationState::failed);
    ASSERT_EQ(poisoned.time().accepted_time, 0.0);
    ASSERT_FALSE(poisoned.advance(0.1).ok());
    ASSERT_TRUE(poisoned.finalize().ok());

    DummyParticipant abandoned;
    Simulation cleanup(abandoned);
    ASSERT_NO_FATAL_FAILURE(prepare(cleanup));
    ASSERT_TRUE(cleanup.begin_window(0.5).ok());
    ASSERT_TRUE(cleanup.advance_trial().ok());
    ASSERT_TRUE(cleanup.finalize().ok());
    ASSERT_EQ(abandoned.state.value, 0.0);
    ASSERT_EQ(abandoned.published_outputs, 0);
    ASSERT_GE(abandoned.calls.size(), 2u);
    ASSERT_EQ(abandoned.calls[abandoned.calls.size() - 2], "restore");
    ASSERT_EQ(abandoned.calls.back(), "finalize");

    DummyParticipant cleanup_failure;
    Simulation retry_cleanup(cleanup_failure);
    ASSERT_NO_FATAL_FAILURE(prepare(retry_cleanup));
    cleanup_failure.failure = Failure::finalize;
    ASSERT_FALSE(retry_cleanup.finalize().ok());
    ASSERT_EQ(retry_cleanup.state(), SimulationState::failed);
    ASSERT_TRUE(retry_cleanup.finalize().ok());

    for (auto failure : {Failure::initialize, Failure::prepare}) {
        DummyParticipant broken;
        Simulation failed(broken);
        broken.failure = failure;
        if (failure == Failure::prepare) {
            ASSERT_TRUE(failed.initialize().ok());
            ASSERT_FALSE(failed.prepare().ok());
        } else {
            ASSERT_FALSE(failed.initialize().ok());
        }
        ASSERT_EQ(failed.state(), SimulationState::failed);
        ASSERT_FALSE(failed.advance(0.1).ok());
        ASSERT_TRUE(failed.finalize().ok());
    }
}

TEST(SimulationTest, ConstraintsAndMetadata) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (double step : {0.0, -1.0, nan, infinity, 1.5}) {
        DummyParticipant participant;
        Simulation simulation(participant);
        ASSERT_NO_FATAL_FAILURE(prepare(simulation));
        ASSERT_FALSE(simulation.advance(step).ok());
        ASSERT_EQ(participant.calls.size(), 2);
        ASSERT_EQ(simulation.time().trial_attempts, 0);
        ASSERT_TRUE(simulation.finalize().ok());
    }
    ASSERT_FALSE(validate_time_constraint({0.0, 0.0, {}}).ok());
    ASSERT_FALSE(validate_time_constraint({1.0, 2.0, {}}).ok());
    ASSERT_FALSE(validate_time_constraint({1.0, 0.0, 0.0}).ok());
    ASSERT_FALSE(validate_time_constraint({1.0, nan, {}}).ok());
    ASSERT_FALSE(validate_time_constraint({1.0, 0.0, infinity}).ok());
    ASSERT_FALSE(validate_time_step({1.0, 0.2, {}}, 0.1).ok());
    ASSERT_FALSE(validate_time_step({1.0, 0.0, 0.1}, 0.15).ok());
    ASSERT_TRUE(validate_time_step({1.0, 0.0, 0.1}, 0.3).ok());
    ASSERT_TRUE(validate_time_step({1e15 + 1.0, 0.0, 1.0}, 1e15).ok());
    ASSERT_FALSE(validate_time_step({1e15 + 1.0, 0.0, 1.0}, 1e15 + 0.5).ok());
    ASSERT_FALSE(validate_time_step({1e15 + 1.0, 0.0, 1.0}, 1e15 + 0.25).ok());

    DummyParticipant quantized;
    quantized.constraint = {1.0, 0.1, 0.1};
    Simulation simulation(quantized);
    ASSERT_NO_FATAL_FAILURE(prepare(simulation));
    ASSERT_FALSE(simulation.advance(0.15).ok());
    ASSERT_TRUE(simulation.advance(0.3).ok());
    ASSERT_TRUE(simulation.finalize().ok());

    DummyParticipant decimal;
    decimal.constraint = {0.1, 0.1, 0.1};
    Simulation repeating(decimal);
    ASSERT_NO_FATAL_FAILURE(prepare(repeating));
    for (int index = 0; index < 1000; ++index) {
        ASSERT_TRUE(repeating.advance(0.1).ok());
    }
    ASSERT_EQ(repeating.time().accepted_steps, 1000);
    ASSERT_TRUE(std::abs(repeating.time().accepted_time - 100.0) < 1e-10);
    ASSERT_TRUE(repeating.finalize().ok());

    for (int invalid = 0; invalid < 3; ++invalid) {
        DummyParticipant participant;
        participant.rollback = invalid != 0;
        participant.invalid_port = invalid == 1;
        participant.duplicate_port = invalid == 2;
        Simulation rejected(participant);
        ASSERT_FALSE(rejected.initialize().ok());
        ASSERT_TRUE(participant.calls.empty());
    }
    DummyParticipant nonfinite;
    Simulation bad_time(nonfinite, nan);
    ASSERT_FALSE(bad_time.initialize().ok());
    ASSERT_TRUE(nonfinite.calls.empty());

    DummyParticipant imprecise;
    Simulation too_small(imprecise, 0x1p54);
    ASSERT_NO_FATAL_FAILURE(prepare(too_small));
    ASSERT_FALSE(too_small.advance(1.0).ok());
    ASSERT_EQ(imprecise.calls.size(), 2);
    ASSERT_TRUE(too_small.finalize().ok());

    DummyParticipant rounded;
    rounded.constraint = {0.75, 0.75, 0.75};
    Simulation rounded_time(rounded, 0x1p52);
    ASSERT_NO_FATAL_FAILURE(prepare(rounded_time));
    ASSERT_TRUE(rounded_time.advance(0.75).ok());
    ASSERT_EQ(rounded_time.time().accepted_time, 0x1p52 + 1.0);
    ASSERT_EQ(rounded.state.value, 0.75);
    ASSERT_TRUE(rounded_time.finalize().ok());
}

} // namespace
