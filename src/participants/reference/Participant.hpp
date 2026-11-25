#pragma once
#include <data/FieldIssuer.hpp>
#include <methods/reference/State.hpp>
#include <nexora/participant/ParticipantIF.hpp>
#include <memory>
#include <optional>
#include <participants/ResourceBudget.hpp>

namespace nexora::reference {
using detail::ResourceBudget;

class Participant final : public ParticipantIF {
  public:
    explicit Participant(Config config = {}, ParticipantId id = ParticipantId{1},
                         ResourceBudget::Limits limits = {});
    // Preflight keeps the validated inputs for execution and needs no Kokkos runtime.
    static Result<std::unique_ptr<Participant>> prepare_case(Config config, ConfigField& invalid_field);
    static Result<TimeConstraint> case_time_constraint(real lambda);
    ParticipantId id() const noexcept override;
    std::vector<PortDescriptor> ports() const override;
    TimeConstraint time_constraint() const override;
    Status initialize(double initial_time) override;
    Status finalize() override;
    Result<Snapshot> snapshot() const;
    ResourceBudget::Usage resource_usage() const { return budget_.usage(); }

  private:
    Status initialize_at_finite_time(double initial_time) override;
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
    TimeConstraint constraint_;
    std::optional<AllocationSize> initial_sizes_;
    std::optional<BankAccess> accepted_access_;
};
} // namespace nexora::reference
