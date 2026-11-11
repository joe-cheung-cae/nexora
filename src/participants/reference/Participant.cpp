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
auto read_field(const detail::FieldIssuer& issuer, const View& view, FieldDescriptor descriptor) {
    const std::array axes{detail::FieldAxis{view.extent(0)}};
    return issuer.read(view, std::move(descriptor), issuer.stamp(), {Association::node, "1", axes});
}

auto read_ids(const detail::FieldIssuer& issuer, const Kokkos::View<const std::uint64_t*>& ids,
              FieldDescriptor descriptor) {
    descriptor.id = FieldId{3};
    descriptor.name = "stable_ids";
    descriptor.scalar_type = ScalarType::uint64;
    descriptor.access = FieldAccess::read;
    return read_field(issuer, ids, std::move(descriptor));
}

auto write_field(const detail::FieldIssuer& issuer, const Values& view, FieldDescriptor descriptor) {
    const std::array axes{detail::FieldAxis{view.extent(0)}};
    return issuer.write(view, std::move(descriptor), issuer.stamp(), {Association::node, "1", axes},
                        FieldAccess::overwrite);
}
} // namespace

template <class View>
Result<Participant::BankAccess> Participant::claim_bank(const View& x, const View& q,
                                                        const FieldStamp& stamp) {
    auto x_access = detail::FieldIssuer::create(x, stamp);
    if (!x_access.ok())
        return x_access.status();
    auto q_access = detail::FieldIssuer::create(q, stamp);
    if (!q_access.ok())
        return q_access.status();
    return BankAccess{std::move(x_access).value(), std::move(q_access).value()};
}

void Participant::discard_window() noexcept {
    trial_ids_access_.reset();
    trial_access_.reset();
    pending_access_.reset();
}

Participant::Participant(Config config, ParticipantId id) : id_(id), state_(std::move(config)) {}
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
TimeConstraint Participant::time_constraint(double) const {
    const real lambda = state_.lambda();
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
    return guarded([&] {
        if (auto status = state_.initialize(t); !status.ok())
            return status;
        auto access = claim_bank(state_.accepted_x_, state_.accepted_q_, state_.accepted_stamp());
        if (!access.ok())
            return access.status();
        accepted_access_.emplace(std::move(access).value());
        return Status{};
    });
}
Status Participant::prepare() {
    return guarded([&] {
        if (!accepted_access_)
            return Status{StatusCode::invalid_state, "reference accepted fields are unavailable"};
        return state_.prepare();
    });
}
Status Participant::begin_window(const TimeInterval& i) {
    if (auto status = guarded([&] { return state_.begin(i); }); !status.ok())
        return status;
    auto status = guarded([&] {
        const auto stamp = state_.trial_stamp();
        auto trial = claim_bank(state_.trial_x_, state_.trial_q_, stamp);
        if (!trial.ok())
            return trial.status();
        auto ids = detail::FieldIssuer::create(state_.ids_, stamp);
        if (!ids.ok())
            return ids.status();
        trial_access_.emplace(std::move(trial).value());
        trial_ids_access_.emplace(std::move(ids).value());
        return Status{};
    });
    if (!status.ok()) {
        discard_window();
        // State::begin succeeded, so this newly opened window is restorable.
        state_.restore();
    }
    return status;
}
Result<TrialReport> Participant::advance_trial(const TimeInterval& i) {
    auto status = guarded([&] {
        if (!trial_access_ || state_.phase_ != State::Phase::open)
            return Status{StatusCode::invalid_state, "advance requires an open reference window"};
        auto x = read_field(accepted_access_->x, state_.accepted_x_,
                            State::field_descriptor(TrialField::x, true, FieldAccess::read));
        if (!x.ok())
            return x.status();
        auto q = read_field(accepted_access_->q, state_.accepted_q_,
                            State::field_descriptor(TrialField::q, true, FieldAccess::read));
        if (!q.ok())
            return q.status();
        auto tx = write_field(trial_access_->x, state_.trial_x_,
                              State::field_descriptor(TrialField::x, false, FieldAccess::overwrite));
        if (!tx.ok())
            return tx.status();
        auto tq = write_field(trial_access_->q, state_.trial_q_,
                              State::field_descriptor(TrialField::q, false, FieldAccess::overwrite));
        if (!tq.ok())
            return tq.status();
        const Kokkos::Serial exec;
        return state_.advance(i, exec, x.value().view(), q.value().view(), tx.value().view(),
                              tq.value().view());
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

Result<TransportView<real>> Participant::send_trial(TrialField field) const {
    if (state_.phase_ != State::Phase::advanced || !trial_access_)
        return Status{StatusCode::invalid_state, "send requires an advanced reference trial"};
    if (field != TrialField::x && field != TrialField::q)
        return Status{StatusCode::invalid_argument, "invalid reference trial field"};

    return guarded([&]() -> Result<TransportView<real>> {
        const bool x = field == TrialField::x;
        auto descriptor = State::field_descriptor(field, false, FieldAccess::read);
        auto values = read_field(x ? trial_access_->x : trial_access_->q,
                                 x ? state_.trial_x_ : state_.trial_q_, descriptor);
        if (!values.ok())
            return values.status();
        auto ids = read_ids(*trial_ids_access_, state_.ids_, std::move(descriptor));
        if (!ids.ok())
            return ids.status();
        const Kokkos::Serial exec;
        // Consumers own the packed copy. They must not prevent precommit
        // from closing the trial authority and registering accepted storage.
        return detail::TransportBuilder::prepare(
            exec, TransportPurpose::coupling, std::move(values).value(), std::move(ids).value(),
            detail::TransportBuilder::SourceLifetime::until_packed);
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
        auto target = write_field(x ? trial_access_->x : trial_access_->q,
                                  x ? state_.trial_x_ : state_.trial_q_,
                                  State::field_descriptor(field, false, FieldAccess::overwrite));
        if (!target.ok())
            return target.status();
        auto ids = read_ids(*trial_ids_access_, state_.ids_,
                            State::field_descriptor(field, false, FieldAccess::read));
        if (!ids.ok())
            return ids.status();
        const Kokkos::Serial exec;
        if (auto received = detail::TransportReceiver::unpack(exec, input, target.value(),
                                                               ids.value(), expected_source);
            !received.ok())
            return received;
        return State::check_finite(target.value().view());
    });
    if (status.ok())
        state_.phase_ = State::Phase::advanced;
    return status;
}

Status Participant::prepare_commit(const TimeInterval& i) {
    if (auto status = guarded([&] { return state_.check_commit(i); }); !status.ok())
        return status;

    // All trial operations are synchronous, with no escaping write grant.
    // Finish any fallible read-only registration before the publish step.
    state_.phase_ = State::Phase::failed_trial;
    discard_window();
    auto status = guarded([&] {
        auto access = claim_bank(ConstValues(state_.trial_x_), ConstValues(state_.trial_q_),
                                 state_.next_accepted_stamp());
        if (!access.ok())
            return access.status();
        pending_access_.emplace(std::move(access).value());
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
        }
        return status;
    });
}
Status Participant::restore_window() {
    return guarded([&] {
        auto status = state_.restore();
        if (status.ok())
            discard_window();
        return status;
    });
}
Status Participant::finalize() {
    return guarded([&] {
        auto status = state_.finalize();
        if (status.ok()) {
            discard_window();
            accepted_access_.reset();
        }
        return status;
    });
}
Result<Snapshot> Participant::snapshot() const {
    try {
        if (!accepted_access_)
            return Status{StatusCode::invalid_state, "reference accepted fields are unavailable"};
        auto result = state_.snapshot();
        if (!result.ok())
            return result.status();
        auto& out = result.value();
        auto x = read_field(accepted_access_->x, state_.accepted_x_, out.x_descriptor());
        if (!x.ok())
            return x.status();
        auto q = read_field(accepted_access_->q, state_.accepted_q_, out.q_descriptor());
        if (!q.ok())
            return q.status();
        out.x = x.value().view();
        out.q = q.value().view();
        out.grants_ = std::make_shared<const Snapshot::Grants>(
            Snapshot::Grants{std::move(x).value(), std::move(q).value()});
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
