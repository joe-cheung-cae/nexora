#include <participants/reference/Participant.hpp>

#include <algorithm>
#include <array>
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

Participant::Participant(Config config, ParticipantId id, ResourceBudget::Limits limits)
    : id_(id), budget_(limits), state_(std::move(config)) {}
ParticipantId Participant::id() const noexcept {
    return id_;
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
    // The default case needs IDs plus accepted and temporary step banks,
    // and only requests its read snapshot after the final step.
    ResourceBudget budget;
    const auto sizes = checked.value();
    auto ids = budget.reserve(sizes.ids_bytes, 0);
    if (!ids.ok())
        return ids.status();
    auto accepted = budget.reserve(sizes.bank_bytes, 0);
    if (!accepted.ok())
        return accepted.status();
    auto next = budget.reserve(sizes.bank_bytes, 1);
    return next.status();
}
Status Participant::validate_case_interval(const TimeInterval& interval, real lambda) {
    return validate_interval(interval, lambda);
}
TimeConstraint Participant::time_constraint() const {
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
Status Participant::do_step(const TimeInterval& interval) {
    return guarded([&] {
        if (!exec_ || !accepted_access_)
            return Status{StatusCode::invalid_state, "step requires initialized reference state"};
        const auto size = state_.preflight_step(interval);
        if (!size.ok())
            return size.status();
        auto reservation = budget_.reserve(size.value());
        if (!reservation.ok())
            return reservation.status();
        Values x(Kokkos::view_alloc(*exec_, Kokkos::WithoutInitializing, "reference.next.x"),
                 state_.accepted_x_.extent(0));
        Values q(Kokkos::view_alloc(*exec_, Kokkos::WithoutInitializing, "reference.next.q"),
                 state_.accepted_q_.extent(0));
        History history(Kokkos::view_alloc(*exec_, Kokkos::WithoutInitializing,
                                           "reference.next.history"), 1);
        const auto next_stamp = Snapshot::accepted_stamp(StateVersion{state_.version_ + 1}, interval.end);
        {
            auto writable_stamp = next_stamp;
            writable_stamp.accepted = false;
            auto candidate = claim_bank(x, q, history, writable_stamp);
            if (!candidate.ok())
                return candidate.status();
            auto& access = candidate.value();
            auto fields = detail::FieldIssuer::grant(
                request_read(accepted_access_->x, state_.accepted_x_,
                             State::field_descriptor(Field::x, FieldAccess::read)),
                request_read(accepted_access_->q, state_.accepted_q_,
                             State::field_descriptor(Field::q, FieldAccess::read)),
                request_read(accepted_access_->history, state_.accepted_history_,
                             State::history_descriptor(FieldAccess::read)),
                request_write(access.x, x, State::field_descriptor(Field::x, FieldAccess::overwrite)),
                request_write(access.q, q, State::field_descriptor(Field::q, FieldAccess::overwrite)),
                request_write(access.history, history, State::history_descriptor(FieldAccess::accumulate)));
            if (!fields.ok())
                return fields.status();
            const auto& [old_x, old_q, old_history, next_x, next_q, next_history] = fields.value();
            Kokkos::deep_copy(*exec_, next_history.view(), old_history.view());
            if (auto status = State::evaluate(*exec_, static_cast<real>(interval.duration()),
                                              state_.lambda(), old_x.view(), old_q.view(),
                                              next_x.view(), next_q.view(), next_history.view(),
                                              state_.accepted_control_.sequence + 1); !status.ok())
                return status;
        }
        // Register read-only authority before the only publication point.
        auto access = claim_bank(ConstValues(x), ConstValues(q), ConstHistory(history), next_stamp);
        if (!access.ok())
            return access.status();
        state_.accepted_x_ = x;
        state_.accepted_q_ = q;
        state_.accepted_history_ = history;
        ++state_.accepted_control_.sequence;
        state_.accepted_diagnostic_ = history(0) + state_.accepted_control_.sequence;
        state_.time_ = interval.end;
        ++state_.version_;
        accepted_access_.emplace(std::move(access).value());
        accepted_budget_ = std::move(reservation).value();
        return Status{};
    });
}

Status Participant::finalize() {
    return guarded([&] {
        if (accepted_access_)
            accepted_access_->close();
        if (exec_)
            exec_->fence("reference close completion");
        auto status = state_.finalize();
        if (status.ok()) {
            accepted_access_.reset();
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
                         State::history_descriptor(FieldAccess::read)));
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
