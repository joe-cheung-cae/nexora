#pragma once
#include <methods/reference/State.hpp>
#include <nexora/participant/ParticipantIF.hpp>

namespace nexora::reference {
class Participant final : public ParticipantIF {
  public:
    explicit Participant(Config config = {}, ParticipantId id = ParticipantId{1});
    ParticipantId id() const noexcept override;
    ParticipantCapabilities capabilities() const override;
    std::vector<PortDescriptor> ports() const override;
    TimeConstraint time_constraint(double accepted_time) const override;
    Status initialize(double initial_time) override;
    Status prepare() override;
    Status begin_window(const TimeInterval& interval) override;
    Result<TrialReport> advance_trial(const TimeInterval& interval) override;
    Status prepare_commit(const TimeInterval& interval) override;
    Status commit_window(const TimeInterval& interval) override;
    Status restore_window() override;
    Status finalize() override;
    Result<Snapshot> snapshot() const;

  private:
    ParticipantId id_;
    State state_;
};
} // namespace nexora::reference
