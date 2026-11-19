#pragma once
#include <data/FieldIssuer.hpp>
#include <methods/reference/State.hpp>
#include <nexora/data/TransportView.hpp>
#include <nexora/participant/ParticipantIF.hpp>
#include <optional>
#include <participants/ResourceBudget.hpp>

namespace nexora::reference {
using detail::ResourceBudget;

class Participant final : public ParticipantIF {
  public:
    explicit Participant(Config config = {}, ParticipantId id = ParticipantId{1},
                         ResourceBudget::Limits limits = {});
    // Application preflight stays at the participant boundary and needs no runtime.
    static Status validate_case_config(const Config& config, double initial_time);
    static Status validate_case_interval(const TimeInterval& interval, real lambda);
    static TimeConstraint case_time_constraint(real lambda);
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
    ResourceBudget::Usage resource_usage() const { return budget_.usage(); }

    enum class Fault : unsigned {
        initialize = 1U << 0, begin = 1U << 1, advance = 1U << 2,
        receive = 1U << 3, prepare = 1U << 4, restore = 1U << 5
    };
    void fail_next(Fault fault) noexcept { faults_ |= static_cast<unsigned>(fault); }
    Result<DiagnosticKey> trial_diagnostic_key() const;
    Result<std::uint64_t> trial_diagnostic(const DiagnosticKey&) const;

    // Local fixture input replaces one trial field after integration and before
    // prepare_commit. A failed receive makes the attempt noncommittable.
    Result<FieldStamp> trial_input_stamp() const;
    Result<TransportView<real>> send_trial(TrialField field) const;
    Status receive_trial(TrialField field, const TransportView<real>& input,
                         const FieldStamp& expected_source);

  private:
    struct BankAccess {
        detail::FieldIssuer x, q, history;
        void close() noexcept;
    };
    template <class View, class HistoryView>
    static Result<BankAccess> claim_bank(const View& x, const View& q, const HistoryView& history,
                                         const FieldStamp& stamp);
    bool consume_fault(Fault) noexcept;
    void close_window() noexcept;
    void discard_window() noexcept;

    ParticipantId id_;
    ResourceBudget budget_;
    ResourceBudget::Ticket ids_budget_, accepted_budget_, trial_budget_;
    std::optional<Kokkos::Serial> exec_;
    State state_;
    std::optional<BankAccess> accepted_access_, trial_access_, pending_access_;
    std::optional<detail::FieldIssuer> trial_ids_access_;
    unsigned faults_{};
};
} // namespace nexora::reference
