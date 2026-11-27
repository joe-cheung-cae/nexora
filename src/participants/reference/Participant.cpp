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
    auto x_access = detail::FieldIssuer::create_at_validated_time(x, stamp);
    if (!x_access.ok())
        return x_access.status();
    auto q_access = detail::FieldIssuer::create_at_validated_time(q, stamp);
    if (!q_access.ok())
        return q_access.status();
    auto history_access = detail::FieldIssuer::create_at_validated_time(history, stamp);
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
    : id_(id), budget_(limits), state_(std::move(config)) {
    const auto constraint = case_time_constraint(state_.lambda());
    constraint_ = constraint.ok() ? constraint.value() : TimeConstraint{.available = false};
}
ParticipantId Participant::id() const noexcept {
    return id_;
}
// The reference fixture exposes accepted output through snapshot(); it does not
// implement a coupling port.
std::vector<PortDescriptor> Participant::ports() const {
    return {};
}
Result<std::unique_ptr<Participant>> Participant::prepare_case(Config config,
                                                               ConfigField& invalid_field) {
    auto participant = std::make_unique<Participant>(std::move(config));
    if (!participant->constraint_.available) {
        invalid_field = ConfigField::lambda;
        return Status{StatusCode::invalid_argument, "invalid reference initial state or lambda"};
    }
    invalid_field = ConfigField::ids;
    const auto checked = validate_initial_fields(participant->state_.config_, &invalid_field);
    if (!checked.ok())
        return checked.status();
    // The default case needs IDs plus accepted and temporary step banks,
    // and only requests its read snapshot after the final step.
    const auto sizes = checked.value();
    auto ids = participant->budget_.reserve(sizes.ids_bytes);
    if (!ids.ok())
        return ids.status();
    auto accepted = participant->budget_.reserve(sizes.bank_bytes);
    if (!accepted.ok())
        return accepted.status();
    auto next = participant->budget_.reserve(sizes.bank_bytes, 1);
    if (!next.ok())
        return next.status();
    participant->initial_sizes_ = sizes;
    return participant;
}
TimeConstraint Participant::time_constraint() const {
    return constraint_;
}
Result<TimeConstraint> Participant::case_time_constraint(real lambda) {
    constexpr auto real_max = std::numeric_limits<real>::max();
    if (!(real{0} < lambda && lambda <= real_max)) {
        return Status{StatusCode::invalid_argument, "invalid reference initial state or lambda"};
    }
    const auto stable_maximum = [&](double step) {
        if constexpr (std::is_same_v<real, float>) {
            if (static_cast<real>(step) * lambda > real{1})
                return false;
        }
        return step * lambda <= 1.0;
    };
    // Cap the step before dividing when the reciprocal would exceed the real range.
    constexpr double step_limit = static_cast<double>(real_max);
    double max_step = lambda <= 1.0 / step_limit ? step_limit : 1.0 / lambda;
    constexpr auto invalid_bounds = "reference lambda has no supported step bounds";
    // Rounding to real can cross the upper stability limit, especially for
    // subnormals. A predecessor in real is below the original double bound.
    if (!stable_maximum(max_step)) {
        max_step = static_cast<double>(std::nextafter(static_cast<real>(max_step), real{0}));
        if (!stable_maximum(max_step))
            return Status{StatusCode::invalid_argument, invalid_bounds};
    }
    const double real_min_step = static_cast<double>(std::numeric_limits<real>::denorm_min());
    const double min_step = std::max(real_min_step, real_min_step / lambda);
    // The cast and multiplication by a positive lambda are monotone. Checking
    // the real product here covers positive dt and factor at every admitted
    // duration, including subnormals. In float the double product cannot
    // underflow if the real product is positive. In double they are identical.
    if (!(static_cast<real>(min_step) * lambda > real{0}))
        return Status{StatusCode::invalid_argument, invalid_bounds};
    return TimeConstraint{max_step, min_step, std::nullopt};
}
Status Participant::initialize(double t) {
    if (std::isfinite(t))
        return initialize_at_finite_time(t);
    return guarded([&] {
        if (auto status = state_.preflight_initialize(); !status.ok())
            return status;
        return Status{StatusCode::invalid_argument, "invalid reference initial state or lambda"};
    });
}
Status Participant::initialize_at_finite_time(double t) {
    ResourceBudget::Ticket ids_ticket, bank_ticket;
    // Keep the selected instance alive until any failed initialization has
    // released its Views and authorities.
    std::optional<Kokkos::Serial> selected;
    auto result = guarded([&] {
        if (auto status = state_.preflight_initialize(); !status.ok())
            return status;
        if (!constraint_.available)
            return Status{StatusCode::invalid_argument, "invalid reference initial state or lambda"};
        if (!initial_sizes_) {
            const auto sizes = validate_initial_fields(state_.config_);
            if (!sizes.ok())
                return sizes.status();
            initial_sizes_ = sizes.value();
        }
        auto ids_reservation = budget_.reserve(initial_sizes_->ids_bytes);
        if (!ids_reservation.ok())
            return ids_reservation.status();
        ids_ticket = std::move(ids_reservation).value();
        auto bank_reservation = budget_.reserve(initial_sizes_->bank_bytes);
        if (!bank_reservation.ok())
            return bank_reservation.status();
        bank_ticket = std::move(bank_reservation).value();

        selected.emplace();
        state_.initialize(t, *selected);
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
            const real factor = static_cast<real>(interval.duration()) * state_.lambda();
            if (auto status = State::evaluate(*exec_, factor, old_x.view(), old_q.view(),
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
        auto out = state_.snapshot();
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
        return out;
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "reference snapshot allocation failed"};
    } catch (const std::exception& e) {
        return Status{StatusCode::execution_failure, e.what()};
    } catch (...) {
        return Status{StatusCode::execution_failure, "reference snapshot failed"};
    }
}
} // namespace nexora::reference
