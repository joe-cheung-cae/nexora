#include <participants/reference/Participant.hpp>

#include <cmath>
#include <exception>
#include <limits>
#include <new>
#include <utility>

namespace nexora::reference {
namespace {
// The selected Serial execution is synchronous. Host exceptions are reported;
// fatal Kokkos/runtime aborts are not recoverable through Status.
template <class Function> Status guarded(Function&& f) {
    try {
        return f();
    } catch (const std::bad_alloc&) {
        return {StatusCode::resource_exhausted, "reference allocation failed"};
    } catch (const std::exception& e) {
        return {StatusCode::execution_failure, e.what()};
    } catch (...) {
        return {StatusCode::execution_failure, "reference operation failed"};
    }
}
} // namespace
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
    const double lambda = state_.lambda();
    if (!std::isfinite(lambda) || lambda <= 0) {
        return {1.0, 0.0, std::nullopt};
    }
    const double max_step = 1.0 / lambda;
    return {std::isfinite(max_step) ? max_step : std::numeric_limits<double>::max(), 0.0,
            std::nullopt};
}
Status Participant::initialize(double t) {
    return guarded([&] { return state_.initialize(t); });
}
Status Participant::prepare() {
    return guarded([&] { return state_.prepare(); });
}
Status Participant::begin_window(const TimeInterval& i) {
    return guarded([&] { return state_.begin(i); });
}
Result<TrialReport> Participant::advance_trial(const TimeInterval& i) {
    auto status = guarded([&] { return state_.advance(i); });
    if (!status.ok()) {
        return status;
    }
    return TrialReport{i.end, true};
}
Status Participant::prepare_commit(const TimeInterval& i) {
    return guarded([&] { return state_.prepare_commit(i); });
}
Status Participant::commit_window(const TimeInterval& i) {
    return guarded([&] { return state_.commit(i); });
}
Status Participant::restore_window() {
    return guarded([&] { return state_.restore(); });
}
Status Participant::finalize() {
    return guarded([&] { return state_.finalize(); });
}
Result<Snapshot> Participant::snapshot() const {
    try {
        return state_.snapshot();
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "reference snapshot allocation failed"};
    } catch (const std::exception& e) {
        return Status{StatusCode::execution_failure, e.what()};
    } catch (...) {
        return Status{StatusCode::execution_failure, "reference snapshot failed"};
    }
}
} // namespace nexora::reference
