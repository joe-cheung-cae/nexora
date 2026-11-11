#include <participants/reference/Participant.hpp>

#include <algorithm>
#include <array>
#include <data/TransportBuilder.hpp>
#include <data/TransportReceiver.hpp>

#include <cmath>
#include <exception>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace nexora::reference {
namespace {
// The selected Serial execution is synchronous. Host exceptions are reported;
// fatal Kokkos/runtime aborts are not recoverable through Status.
template <class Function> auto guarded(Function&& f) -> std::invoke_result_t<Function> {
    try {
        return f();
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "reference allocation failed"};
    } catch (const std::exception& e) {
        return Status{StatusCode::execution_failure, e.what()};
    } catch (...) {
        return Status{StatusCode::execution_failure, "reference operation failed"};
    }
}
template <class View>
auto request_read(const detail::FieldIssuer& issuer, const View& view, FieldDescriptor descriptor) {
    const std::array axes{detail::FieldAxis{view.extent(0)}};
    return issuer.request_read(view, std::move(descriptor), issuer.stamp(), {Association::node, "1", axes});
}

auto request_ids(const detail::FieldIssuer& issuer, const ConstEntityIds& ids,
                 FieldDescriptor descriptor) {
    descriptor.id = FieldId{3};
    descriptor.name = "stable_ids";
    descriptor.scalar_type = ScalarType::uint64;
    descriptor.access = FieldAccess::read;
    return request_read(issuer, ids, std::move(descriptor));
}

template <class View>
auto request_write(const detail::FieldIssuer& issuer, const View& view, FieldDescriptor descriptor) {
    const std::array axes{detail::FieldAxis{view.extent(0)}};
    const auto access = descriptor.access;
    return issuer.request_write(view, std::move(descriptor), issuer.stamp(),
                                {Association::node, "1", axes}, access);
}
} // namespace

template <class View, class HistoryView>
Result<Participant::BankAccess> Participant::claim_bank(const View& x, const View& q,
                                                        const HistoryView& history,
                                                        const FieldStamp& stamp) {
    auto x_access = detail::FieldIssuer::create(x, stamp);
    if (!x_access.ok())
        return x_access.status();
    auto q_access = detail::FieldIssuer::create(q, stamp);
    if (!q_access.ok())
        return q_access.status();
    auto history_access = detail::FieldIssuer::create(history, stamp);
    if (!history_access.ok())
        return history_access.status();
    return BankAccess{std::move(x_access).value(), std::move(q_access).value(),
                      std::move(history_access).value()};
}

void Participant::BankAccess::close() noexcept {
    x.close();
    q.close();
    history.close();
}

bool Participant::consume_fault(Fault fault) noexcept {
    const auto bit = static_cast<unsigned>(fault);
    const bool armed = (faults_ & bit) != 0;
    faults_ &= ~bit;
    return armed;
}

void Participant::close_window() noexcept {
    if (trial_ids_access_)
        trial_ids_access_->close();
    if (trial_access_)
        trial_access_->close();
    if (pending_access_)
        pending_access_->close();
}

void Participant::discard_window() noexcept {
    trial_ids_access_.reset();
    trial_access_.reset();
    pending_access_.reset();
}

Participant::Participant(Config config, ParticipantId id, ResourceBudget::Limits limits)
    : id_(id), budget_(limits), state_(std::move(config)) {}
ParticipantId Participant::id() const noexcept {
    return id_;
}
ParticipantCapabilities Participant::capabilities() const {
    return {true};
}
// The reference fixture exposes accepted output through snapshot(); it does not
// implement a coupling port.
std::vector<PortDescriptor> Participant::ports() const {
    return {};
}
Status Participant::validate_case_config(const Config& config, double initial_time) {
    const auto checked = validate_config(config, initial_time);
    if (!checked.ok())
        return checked.status();
    // The default case needs IDs plus accepted/trial banks at a window boundary,
    // and only requests its read snapshot after the final commit.
    ResourceBudget budget;
    const auto sizes = checked.value();
    auto ids = budget.reserve(sizes.ids_bytes, 0);
    if (!ids.ok())
        return ids.status();
    auto accepted = budget.reserve(sizes.bank_bytes, 0);
    if (!accepted.ok())
        return accepted.status();
    auto trial = budget.reserve(sizes.bank_bytes, 1);
    return trial.status();
}
Status Participant::validate_case_interval(const TimeInterval& interval, real lambda) {
    return validate_interval(interval, lambda);
}
TimeConstraint Participant::time_constraint(double) const {
    return case_time_constraint(state_.lambda());
}
TimeConstraint Participant::case_time_constraint(real lambda) {
    if (!std::isfinite(lambda) || lambda <= 0) {
        return {1.0, 0.0, std::nullopt};
    }
    const double lambda_time = static_cast<double>(lambda);
    double max_step = std::min(1.0 / lambda_time,
                               static_cast<double>(std::numeric_limits<real>::max()));
    const real real_max_step = static_cast<real>(max_step);
    // Rounding the time bound to real can cross the stable-step boundary,
    // especially when real_max_step is subnormal. Publish a supported endpoint.
    if (max_step * lambda_time > 1.0 || real_max_step * lambda > real{1}) {
        max_step = static_cast<double>(std::nextafter(real_max_step, real{0}));
    }
    const double real_min_step = static_cast<double>(std::numeric_limits<real>::denorm_min());
    const double min_step = std::max(real_min_step, real_min_step / lambda_time);
    return {max_step, min_step, std::nullopt};
}
Status Participant::initialize(double t) {
    ResourceBudget::Ticket ids_ticket, bank_ticket;
    // Keep the selected instance alive until any failed initialization has
    // released its Views and authorities.
    std::optional<Kokkos::Serial> selected;
    auto result = guarded([&] {
        const auto sizes = state_.preflight_initialize(t);
        if (!sizes.ok())
            return sizes.status();
        auto ids_reservation = budget_.reserve(sizes.value().ids_bytes);
        if (!ids_reservation.ok())
            return ids_reservation.status();
        ids_ticket = std::move(ids_reservation).value();
        auto bank_reservation = budget_.reserve(sizes.value().bank_bytes);
        if (!bank_reservation.ok())
            return bank_reservation.status();
        bank_ticket = std::move(bank_reservation).value();

        selected.emplace();
        if (auto status = state_.initialize(t, *selected); !status.ok())
            return status;
        if (consume_fault(Fault::initialize))
            return Status{StatusCode::execution_failure, "injected reference initialization failure"};
        auto access = claim_bank(state_.accepted_x_, state_.accepted_q_, state_.accepted_history_,
                                 state_.accepted_stamp());
        if (!access.ok())
            return access.status();
        exec_.emplace(*selected);
        accepted_access_.emplace(std::move(access).value());
        ids_budget_ = std::move(ids_ticket);
        accepted_budget_ = std::move(bank_ticket);
        state_.finish_initialization();
        return Status{};
    });
    if (!result.ok() && state_.phase_ == State::Phase::initialized && !accepted_access_) {
        state_.cancel_initialization();
        exec_.reset();
    }
    return result;
}
Status Participant::prepare() {
    return guarded([&] {
        if (!accepted_access_)
            return Status{StatusCode::invalid_state, "reference accepted fields are unavailable"};
        return state_.prepare();
    });
}
Status Participant::begin_window(const TimeInterval& i) {
    if (!exec_ || !accepted_access_)
        return {StatusCode::invalid_state, "begin requires prepared reference state"};
    ResourceBudget::Ticket bank_ticket;
    bool opened = false;
    auto status = guarded([&] {
        const auto size = state_.preflight_begin(i);
        if (!size.ok())
            return size.status();
        auto reservation = budget_.reserve(size.value());
        if (!reservation.ok())
            return reservation.status();
        bank_ticket = std::move(reservation).value();
        auto history = detail::FieldIssuer::grant(request_read(
            accepted_access_->history, state_.accepted_history_,
            State::history_descriptor(true, FieldAccess::read)));
        if (!history.ok())
            return history.status();
        if (auto begun = state_.begin(i, *exec_, std::get<0>(history.value()).view()); !begun.ok())
            return begun;
        opened = true;
        const auto stamp = state_.trial_stamp();
        auto trial = claim_bank(state_.trial_x_, state_.trial_q_, state_.trial_history_, stamp);
        if (!trial.ok())
            return trial.status();
        auto ids = detail::FieldIssuer::create(state_.ids_, stamp);
        if (!ids.ok())
            return ids.status();
        if (consume_fault(Fault::begin))
            return Status{StatusCode::execution_failure, "injected reference begin failure"};
        trial_access_.emplace(std::move(trial).value());
        trial_ids_access_.emplace(std::move(ids).value());
        trial_budget_ = std::move(bank_ticket);
        return Status{};
    });
    if (!status.ok() && opened) {
        // State::begin completed a synchronous private candidate copy.
        state_.restore();
        discard_window();
    }
    return status;
}
Result<TrialReport> Participant::advance_trial(const TimeInterval& i) {
    auto status = guarded([&] {
        if (!trial_access_ || state_.phase_ != State::Phase::open)
            return Status{StatusCode::invalid_state, "advance requires an open reference window"};
        // One grant set spans the relaxation kernel and its finite-value check.
        auto fields = detail::FieldIssuer::grant(
            request_read(accepted_access_->x, state_.accepted_x_,
                         State::field_descriptor(TrialField::x, true, FieldAccess::read)),
            request_read(accepted_access_->q, state_.accepted_q_,
                         State::field_descriptor(TrialField::q, true, FieldAccess::read)),
            request_write(trial_access_->x, state_.trial_x_,
                          State::field_descriptor(TrialField::x, false, FieldAccess::overwrite)),
            request_write(trial_access_->q, state_.trial_q_,
                          State::field_descriptor(TrialField::q, false, FieldAccess::overwrite)),
            request_write(trial_access_->history, state_.trial_history_,
                          State::history_descriptor(false, FieldAccess::accumulate)));
        if (!fields.ok())
            return fields.status();
        const auto& [x, q, tx, tq, history] = fields.value();
        auto advanced = state_.advance(i, *exec_, x.view(), q.view(), tx.view(), tq.view(),
                                       history.view(), history.stamp());
        if (advanced.ok() && consume_fault(Fault::advance)) {
            state_.phase_ = State::Phase::failed_trial;
            return Status{StatusCode::execution_failure, "injected reference advance failure"};
        }
        return advanced;
    });
    if (!status.ok())
        return status;
    return TrialReport{i.end, true};
}

Result<FieldStamp> Participant::trial_input_stamp() const {
    if (state_.phase_ != State::Phase::advanced || !trial_access_)
        return Status{StatusCode::invalid_state, "receive requires an advanced reference trial"};
    return state_.trial_stamp();
}

Result<DiagnosticKey> Participant::trial_diagnostic_key() const {
    return state_.trial_diagnostic_key();
}

Result<std::uint64_t> Participant::trial_diagnostic(const DiagnosticKey& key) const {
    return state_.trial_diagnostic(key);
}

Result<TransportView<real>> Participant::send_trial(TrialField field) const {
    if (state_.phase_ != State::Phase::advanced || !trial_access_)
        return Status{StatusCode::invalid_state, "send requires an advanced reference trial"};
    if (field != TrialField::x && field != TrialField::q)
        return Status{StatusCode::invalid_argument, "invalid reference trial field"};

    return guarded([&]() -> Result<TransportView<real>> {
        const auto bytes = detail::TransportBuilder::packed_payload_bytes<real>(state_.ids_.extent(0), 1);
        if (!bytes.ok())
            return bytes.status();
        auto reservation = budget_.reserve(bytes.value(), 1);
        if (!reservation.ok())
            return reservation.status();
        const bool x = field == TrialField::x;
        auto descriptor = State::field_descriptor(field, false, FieldAccess::read);
        auto fields = detail::FieldIssuer::grant(
            request_read(x ? trial_access_->x : trial_access_->q,
                         x ? state_.trial_x_ : state_.trial_q_, descriptor),
            request_ids(*trial_ids_access_, state_.ids_, descriptor));
        if (!fields.ok())
            return fields.status();
        auto& [values, ids] = fields.value();
        // Consumers own the packed copy. They must not prevent precommit
        // from closing the trial authority and registering accepted storage.
        return detail::TransportBuilder::prepare(
            *exec_, TransportPurpose::coupling, std::move(values), std::move(ids),
            detail::TransportBuilder::SourceLifetime::until_packed, std::move(reservation).value());
    });
}

Status Participant::receive_trial(TrialField field, const TransportView<real>& input,
                                  const FieldStamp& expected_source) {
    if (state_.phase_ != State::Phase::advanced || !trial_access_)
        return {StatusCode::invalid_state, "receive requires an advanced reference trial"};

    // A failed receive cannot subsequently publish the original or a partly
    // replaced trial. Simulation can restore it through its existing path.
    state_.phase_ = State::Phase::failed_trial;
    auto status = guarded([&] {
        if ((field != TrialField::x && field != TrialField::q) ||
            input.purpose() != TransportPurpose::coupling)
            return Status{StatusCode::invalid_argument, "invalid reference trial field or purpose"};
        const bool x = field == TrialField::x;
        auto fields = detail::FieldIssuer::grant(
            request_write(x ? trial_access_->x : trial_access_->q,
                          x ? state_.trial_x_ : state_.trial_q_,
                          State::field_descriptor(field, false, FieldAccess::overwrite)),
            request_ids(*trial_ids_access_, state_.ids_,
                        State::field_descriptor(field, false, FieldAccess::read)));
        if (!fields.ok())
            return fields.status();
        const auto& [target, ids] = fields.value();
        if (auto received = detail::TransportReceiver::unpack(*exec_, input, target,
                                                               ids, expected_source);
            !received.ok())
            return received;
        auto finite = State::check_finite(*exec_, target.view());
        if (finite.ok() && consume_fault(Fault::receive))
            return Status{StatusCode::execution_failure, "injected reference receive failure"};
        return finite;
    });
    if (status.ok())
        state_.phase_ = State::Phase::advanced;
    return status;
}

Status Participant::prepare_commit(const TimeInterval& i) {
    if (!trial_access_)
        return {StatusCode::invalid_state, "prepare_commit requires an advanced reference trial"};
    if (auto status = guarded([&] { return state_.check_commit(i, trial_access_->history.stamp()); });
        !status.ok())
        return status;

    // All trial operations are synchronous, with no escaping write grant.
    // Finish any fallible read-only registration before the publish step.
    state_.phase_ = State::Phase::failed_trial;
    discard_window();
    auto status = guarded([&] {
        auto access = claim_bank(ConstValues(state_.trial_x_), ConstValues(state_.trial_q_),
                                 ConstHistory(state_.trial_history_),
                                 state_.next_accepted_stamp());
        if (!access.ok())
            return access.status();
        pending_access_.emplace(std::move(access).value());
        if (consume_fault(Fault::prepare))
            return Status{StatusCode::execution_failure, "injected reference prepare failure"};
        return Status{};
    });
    if (status.ok())
        state_.phase_ = State::Phase::commit_ready;
    return status;
}
Status Participant::commit_window(const TimeInterval& i) {
    return guarded([&] {
        if (!pending_access_)
            return Status{StatusCode::invalid_state, "commit requires prepared reference fields"};
        auto status = state_.commit(i);
        if (status.ok()) {
            accepted_access_.emplace(std::move(*pending_access_));
            pending_access_.reset();
            accepted_budget_ = std::move(trial_budget_);
        }
        return status;
    });
}
Status Participant::restore_window() {
    using Phase = State::Phase;
    const auto phase = state_.phase_;
    const bool open = phase == Phase::open || phase == Phase::advanced ||
                      phase == Phase::failed_trial || phase == Phase::commit_ready;
    auto result = guarded([&] {
        if (open) {
            close_window();
            if (consume_fault(Fault::restore))
                return Status{StatusCode::execution_failure, "injected reference restore failure"};
            exec_->fence("reference restore completion");
        }
        auto status = state_.restore();
        if (status.ok()) {
            discard_window();
            trial_budget_.reset();
        }
        return status;
    });
    if (!result.ok() && open)
        state_.phase_ = Phase::failed;
    return result;
}
Status Participant::finalize() {
    return guarded([&] {
        if (state_.phase_ != State::Phase::finalized) {
            // Close issuance before the producer completion boundary. Detached
            // consumer records keep their existing grants and budget tickets.
            state_.phase_ = State::Phase::failed;
            close_window();
            if (accepted_access_)
                accepted_access_->close();
            if (exec_)
                exec_->fence("reference close completion");
        }
        auto status = state_.finalize();
        if (status.ok()) {
            discard_window();
            accepted_access_.reset();
            trial_budget_.reset();
            accepted_budget_.reset();
            ids_budget_.reset();
            exec_.reset();
        }
        return status;
    });
}
Result<Snapshot> Participant::snapshot() const {
    try {
        if (!accepted_access_)
            return Status{StatusCode::invalid_state, "reference accepted fields are unavailable"};
        auto reservation = budget_.reserve(0, 1);
        if (!reservation.ok())
            return reservation.status();
        auto result = state_.snapshot();
        if (!result.ok())
            return result.status();
        auto& out = result.value();
        auto fields = detail::FieldIssuer::grant(
            request_read(accepted_access_->x, state_.accepted_x_, out.x_descriptor()),
            request_read(accepted_access_->q, state_.accepted_q_, out.q_descriptor()),
            request_read(accepted_access_->history, state_.accepted_history_,
                         State::history_descriptor(true, FieldAccess::read)));
        if (!fields.ok())
            return fields.status();
        auto& [x, q, history] = fields.value();
        out.x = x.view();
        out.q = q.view();
        out.history = history.view();
        out.grants_ = std::make_shared<const Snapshot::Grants>(
            Snapshot::Grants{ids_budget_, accepted_budget_, std::move(reservation).value(),
                             std::move(x), std::move(q), std::move(history)});
        return result;
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "reference snapshot allocation failed"};
    } catch (const std::exception& e) {
        return Status{StatusCode::execution_failure, e.what()};
    } catch (...) {
        return Status{StatusCode::execution_failure, "reference snapshot failed"};
    }
}
} // namespace nexora::reference
