#pragma once
#include <data/FieldIssuer.hpp>
#include <methods/reference/State.hpp>
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
    std::vector<PortDescriptor> ports() const override;
    TimeConstraint time_constraint() const override;
    Status initialize(double initial_time) override;
    Status finalize() override;
    Result<Snapshot> snapshot() const;
    ResourceBudget::Usage resource_usage() const { return budget_.usage(); }

  private:
    Status do_step(const TimeInterval& interval) override;
    struct BankAccess {
        detail::FieldIssuer x, q, history;
        void close() noexcept;
    };
    template <class View, class HistoryView>
    static Result<BankAccess> claim_bank(const View& x, const View& q, const HistoryView& history,
                                         const FieldStamp& stamp);

    ParticipantId id_;
    ResourceBudget budget_;
    ResourceBudget::Ticket ids_budget_, accepted_budget_;
    std::optional<Kokkos::Serial> exec_;
    State state_;
    std::optional<BankAccess> accepted_access_;
};
} // namespace nexora::reference
