#include <nexora/simulation/Simulation.hpp>

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

using namespace nexora;

class TestParticipant final : public ParticipantIF {
  public:
    ParticipantId id() const noexcept override { return ParticipantId{1}; }

    std::vector<PortDescriptor> ports() const override {
        FieldDescriptor field;
        field.id = FieldId{1};
        field.name = "test_value";
        field.units = invalid_port ? "" : "1";
        const PortDescriptor port{PortId{1}, SupportId{1}, PortDirection::output, field};
        return duplicate_port ? std::vector<PortDescriptor>{port, port}
                              : std::vector<PortDescriptor>{port};
    }

    TimeConstraint time_constraint() const override {
        ++constraint_queries;
        return constraint;
    }

    Status initialize(double initial_time) override {
        ++initializations;
        time = initial_time;
        return fail_initialize ? Status{StatusCode::execution_failure, "initialization failed"} : Status{};
    }

    Status finalize() override {
        ++finalizations;
        if (fail_finalize) {
            fail_finalize = false;
            return {StatusCode::execution_failure, "finalization failed"};
        }
        return {};
    }

    TimeConstraint constraint{1.0, 0.0, std::nullopt};
    std::optional<double> next_maximum;
    int initializations = 0;
    int finalizations = 0;
    mutable int constraint_queries = 0;
    int evaluations = 0;
    int publications = 0;
    double time = 0.0;
    double value = 0.0;
    int fail_at = -1;
    bool invalid_port = false;
    bool duplicate_port = false;
    bool fail_initialize = false;
    bool fail_finalize = false;
    Simulation* reenter = nullptr;

  private:
    Status do_step(const TimeInterval& interval) override {
        EXPECT_DOUBLE_EQ(time, interval.begin);
        ++evaluations;
        const double candidate = value + interval.duration() * (publications + 1);
        if (evaluations == fail_at)
            return {StatusCode::numerical_failure, "candidate is invalid"};
        if (reenter) {
            EXPECT_DOUBLE_EQ(reenter->time().accepted_time, interval.begin);
            EXPECT_EQ(reenter->initialize().code(), StatusCode::invalid_state);
            EXPECT_EQ(reenter->run(interval.duration()).code(), StatusCode::invalid_state);
            EXPECT_EQ(reenter->finalize().code(), StatusCode::invalid_state);
        }
        value = candidate;
        time = interval.end;
        ++publications;
        if (next_maximum)
            constraint.maximum_step = *next_maximum;
        return {};
    }
};

TEST(SimulationTest, RunsScheduleAndPublishesEachStepOnce) {
    TestParticipant participant;
    Simulation simulation(participant, 2.0);
    participant.reenter = &simulation;
    const std::array steps{0.5, 0.25, 0.125};
    EXPECT_EQ(simulation.run(steps).code(), StatusCode::invalid_state);
    ASSERT_TRUE(simulation.initialize().ok());
    EXPECT_EQ(simulation.initialize().code(), StatusCode::invalid_state);
    ASSERT_TRUE(simulation.run(steps).ok());
    EXPECT_EQ(simulation.time().accepted_steps, 3);
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 2.875);
    EXPECT_DOUBLE_EQ(participant.value, 1.375);
    EXPECT_EQ(participant.evaluations, 3);
    EXPECT_EQ(participant.publications, 3);
    EXPECT_EQ(participant.constraint_queries, 3);
    EXPECT_EQ(participant.initializations, 1);
    ASSERT_TRUE(simulation.finalize().ok());
    EXPECT_EQ(participant.finalizations, 1);
    EXPECT_EQ(simulation.finalize().code(), StatusCode::invalid_state);
    EXPECT_EQ(simulation.run(0.5).code(), StatusCode::invalid_state);
}

TEST(SimulationTest, FailedStepStopsRunAndPreservesLastPublication) {
    TestParticipant participant;
    participant.fail_at = 2;
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto status = simulation.run(0.5, 3);
    EXPECT_EQ(status.code(), StatusCode::numerical_failure);
    EXPECT_EQ(participant.evaluations, 2);
    EXPECT_EQ(participant.constraint_queries, 2);
    EXPECT_EQ(participant.publications, 1);
    EXPECT_EQ(simulation.time().accepted_steps, 1);
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0.5);
    EXPECT_DOUBLE_EQ(participant.time, 0.5);
    EXPECT_DOUBLE_EQ(participant.value, 0.5);
    EXPECT_EQ(simulation.run(0.25).code(), StatusCode::invalid_state);
    EXPECT_EQ(participant.evaluations, 2);
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(SimulationTest, UsesPublishedConstraintOnTheNextStep) {
    for (double maximum : {0.25, -0.25}) {
        TestParticipant participant;
        participant.next_maximum = maximum;
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        const auto status =
            simulation.run(std::array{0.5, std::nextafter(0.25, 1.0), 0.125});
        EXPECT_EQ(status.code(), StatusCode::invalid_argument);
        EXPECT_EQ(participant.evaluations, 1);
        EXPECT_EQ(participant.constraint_queries, 2);
        EXPECT_EQ(simulation.time().accepted_steps, 1);
        EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0.5);
        ASSERT_TRUE(simulation.finalize().ok());
    }
}

TEST(SimulationTest, ExhaustedTimeDomainStopsRunWithoutAnotherPublication) {
    const std::array<std::array<double, 2>, 3> schedules{
        {{1.0, 0.25}, {1.0, 0x1p-54}, {0x1p1023, std::numeric_limits<double>::max()}}};
    for (const auto& schedule : schedules) {
        const double accepted = schedule[0];
        const double requested = schedule[1];
        TestParticipant participant;
        participant.constraint.maximum_step = accepted;
        participant.next_maximum = 0.0;
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        ASSERT_TRUE(simulation.run(accepted).ok());
        EXPECT_EQ(simulation.time().accepted_steps, 1);
        EXPECT_DOUBLE_EQ(simulation.time().accepted_time, accepted);

        const auto status = simulation.run(requested);
        EXPECT_EQ(status.code(), StatusCode::invalid_argument);
        EXPECT_EQ(status.message(), "participant finite time domain is exhausted");
        EXPECT_EQ(participant.evaluations, 1);
        EXPECT_EQ(participant.publications, 1);
        EXPECT_EQ(participant.constraint_queries, 2);
        EXPECT_EQ(simulation.time().accepted_steps, 1);
        EXPECT_DOUBLE_EQ(simulation.time().accepted_time, accepted);
        EXPECT_DOUBLE_EQ(participant.time, accepted);
        EXPECT_DOUBLE_EQ(participant.value, accepted);
        EXPECT_EQ(simulation.run(0.25).code(), StatusCode::invalid_state);
        EXPECT_EQ(participant.constraint_queries, 2);
        ASSERT_TRUE(simulation.finalize().ok());
        EXPECT_EQ(participant.finalizations, 1);
    }
}

TEST(SimulationTest, ExhaustedTimeDomainPreservesInvalidInputDiagnostics) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    for (double requested : {0.0, -1.0, nan, infinity}) {
        TestParticipant participant;
        participant.constraint.maximum_step = 0.0;
        Simulation simulation(participant);
        ASSERT_TRUE(simulation.initialize().ok());
        const auto status = simulation.run(requested);
        EXPECT_EQ(status.code(), StatusCode::invalid_argument);
        EXPECT_EQ(status.message(), "requested step violates participant bounds");
        EXPECT_EQ(participant.constraint_queries, 1);
        EXPECT_EQ(participant.evaluations, 0);
        EXPECT_EQ(simulation.time().accepted_steps, 0);
        EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0.0);
        ASSERT_TRUE(simulation.finalize().ok());
    }
    const auto invalid_bounds = validate_time_step({0.0, 0.25, {}}, 0.25);
    EXPECT_EQ(invalid_bounds.code(), StatusCode::invalid_argument);
    EXPECT_EQ(invalid_bounds.message(), "invalid participant step bounds");
    const auto invalid_quantum = validate_time_step({0.0, 0.0, nan}, 0.25);
    EXPECT_EQ(invalid_quantum.code(), StatusCode::invalid_argument);
    EXPECT_EQ(invalid_quantum.message(), "fixed time quantum must be finite and positive");
    const auto positive_bound = validate_time_step({0.25, 0.0, {}}, 0.5);
    EXPECT_EQ(positive_bound.code(), StatusCode::invalid_argument);
    EXPECT_EQ(positive_bound.message(), "requested step violates participant bounds");
}

TEST(SimulationTest, UnavailableConstraintIsDistinctFromExhaustedTimeDomain) {
    TestParticipant participant;
    participant.constraint.available = false;
    Simulation simulation(participant);
    ASSERT_TRUE(simulation.initialize().ok());
    const auto status = simulation.run(0.25);
    EXPECT_EQ(status.code(), StatusCode::invalid_state);
    EXPECT_EQ(status.message(), "participant time constraint is unavailable");
    EXPECT_EQ(participant.constraint_queries, 1);
    EXPECT_EQ(participant.evaluations, 0);
    EXPECT_EQ(participant.publications, 0);
    EXPECT_EQ(simulation.time().accepted_steps, 0);
    EXPECT_DOUBLE_EQ(simulation.time().accepted_time, 0.0);
    ASSERT_TRUE(simulation.finalize().ok());
}

TEST(SimulationTest, FinalizesAfterInitializationOrFinalizationFailure) {
    TestParticipant broken;
    broken.fail_initialize = true;
    Simulation failed(broken);
    EXPECT_EQ(failed.initialize().code(), StatusCode::execution_failure);
    EXPECT_EQ(failed.run(0.5).code(), StatusCode::invalid_state);
    ASSERT_TRUE(failed.finalize().ok());
    EXPECT_EQ(broken.finalizations, 1);

    TestParticipant cleanup;
    cleanup.fail_finalize = true;
    Simulation finalize_failed(cleanup);
    ASSERT_TRUE(finalize_failed.initialize().ok());
    EXPECT_EQ(finalize_failed.finalize().code(), StatusCode::execution_failure);
    EXPECT_EQ(finalize_failed.run(0.5).code(), StatusCode::invalid_state);
    ASSERT_TRUE(finalize_failed.finalize().ok());
    EXPECT_EQ(cleanup.finalizations, 2);

    TestParticipant unused;
    Simulation abandoned(unused);
    ASSERT_TRUE(abandoned.finalize().ok());
    EXPECT_EQ(unused.finalizations, 1);
}

TEST(SimulationTest, ConstraintsAndMetadata) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(validate_time_constraint({0.0, 0.0, {}}).ok());
    EXPECT_FALSE(validate_time_constraint({1.0, 2.0, {}}).ok());
    EXPECT_FALSE(validate_time_constraint({1.0, 0.0, 0.0}).ok());
    EXPECT_FALSE(validate_time_constraint({1.0, nan, {}}).ok());
    EXPECT_FALSE(validate_time_constraint({1.0, 0.0, infinity}).ok());
    for (double step : {0.0, -1.0, nan, infinity, std::nextafter(1.0, infinity)})
        EXPECT_FALSE(validate_time_step({}, step).ok());
    EXPECT_FALSE(validate_time_step({1.0, 0.2, {}}, std::nextafter(0.2, 0.0)).ok());
    EXPECT_FALSE(validate_time_step({1.0, 0.0, 0.1}, 0.15).ok());
    EXPECT_TRUE(validate_time_step({1.0, 0.0, 0.1}, 0.3).ok());
    EXPECT_TRUE(validate_time_step({1e15 + 1.0, 0.0, 1.0}, 1e15).ok());
    EXPECT_FALSE(validate_time_step({1e15 + 1.0, 0.0, 1.0}, 1e15 + 0.5).ok());
    EXPECT_FALSE(validate_time_step({1e15 + 1.0, 0.0, 1.0}, 1e15 + 0.25).ok());

    TestParticipant decimal;
    decimal.constraint = {0.1, 0.1, 0.1};
    Simulation repeating(decimal);
    ASSERT_TRUE(repeating.initialize().ok());
    ASSERT_TRUE(repeating.run(0.1, 1000).ok());
    EXPECT_EQ(repeating.time().accepted_steps, 1000);
    EXPECT_NEAR(repeating.time().accepted_time, 100.0, 1e-10);
    ASSERT_TRUE(repeating.finalize().ok());

    for (bool duplicate : {false, true}) {
        TestParticipant participant;
        participant.invalid_port = !duplicate;
        participant.duplicate_port = duplicate;
        Simulation rejected(participant);
        EXPECT_EQ(rejected.initialize().code(), StatusCode::invalid_argument);
        EXPECT_EQ(participant.initializations, 0);
        ASSERT_TRUE(rejected.finalize().ok());
    }
    TestParticipant nonfinite;
    Simulation bad_time(nonfinite, nan);
    EXPECT_EQ(bad_time.initialize().code(), StatusCode::invalid_argument);
    EXPECT_EQ(nonfinite.initializations, 0);
    ASSERT_TRUE(bad_time.finalize().ok());
}

TEST(SimulationTest, IntegratesRequestedDurationWithRoundedEndpoint) {
    TestParticipant imprecise;
    Simulation too_small(imprecise, 0x1p54);
    ASSERT_TRUE(too_small.initialize().ok());
    const auto stalled = too_small.run(1.0);
    EXPECT_EQ(stalled.code(), StatusCode::invalid_argument);
    EXPECT_EQ(stalled.message(), "step does not produce a finite advancing endpoint");
    EXPECT_EQ(imprecise.evaluations, 0);
    EXPECT_EQ(imprecise.constraint_queries, 1);
    ASSERT_TRUE(too_small.finalize().ok());

    TestParticipant rounded;
    rounded.constraint = {0.75, 0.75, 0.75};
    Simulation rounded_time(rounded, 0x1p52);
    ASSERT_TRUE(rounded_time.initialize().ok());
    ASSERT_TRUE(rounded_time.run(0.75).ok());
    EXPECT_DOUBLE_EQ(rounded_time.time().accepted_time, 0x1p52 + 1.0);
    EXPECT_DOUBLE_EQ(rounded.value, 0.75);
    ASSERT_TRUE(rounded_time.finalize().ok());
}

} // namespace
