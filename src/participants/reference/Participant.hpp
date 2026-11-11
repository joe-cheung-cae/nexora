#pragma once
#include <data/FieldIssuer.hpp>
#include <methods/reference/State.hpp>
#include <nexora/data/TransportView.hpp>
#include <nexora/participant/ParticipantIF.hpp>
#include <optional>

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

    // Local fixture input replaces one trial field after integration and before
    // prepare_commit. A failed receive makes the attempt noncommittable.
    Result<FieldStamp> trial_input_stamp() const;
    Result<TransportView<real>> send_trial(TrialField field) const;
    Status receive_trial(TrialField field, const TransportView<real>& input,
                         const FieldStamp& expected_source);

  private:
    struct BankAccess {
        detail::FieldIssuer x, q;
    };
    template <class View>
    static Result<BankAccess> claim_bank(const View& x, const View& q, const FieldStamp& stamp);
    void discard_window() noexcept;

    ParticipantId id_;
    State state_;
    std::optional<BankAccess> accepted_access_, trial_access_, pending_access_;
    std::optional<detail::FieldIssuer> trial_ids_access_;
};
} // namespace nexora::reference
