#include <nexora/simulation/Simulation.hpp>

#include <cmath>
#include <limits>

namespace nexora {
namespace {

Status invalid_state(const char* message) {
    return {StatusCode::invalid_state, message};
}

} // namespace

Simulation::Simulation(ParticipantIF& participant, double initial_time) noexcept
    : participant_(participant) {
    time_.accepted_time = initial_time;
}

Status Simulation::initialize() {
    if (state_ != SimulationState::created) {
        return invalid_state("simulation can only initialize once");
    }
    if (!std::isfinite(time_.accepted_time)) {
        return {StatusCode::invalid_argument, "initial time must be finite"};
    }
    if (!participant_.capabilities().rollback) {
        return {StatusCode::unsupported, "transactional simulation requires participant rollback"};
    }
    const auto ports = participant_.ports();
    for (std::size_t index = 0; index < ports.size(); ++index) {
        const auto status = ports[index].validate();
        if (!status.ok()) {
            return status;
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (ports[previous].id == ports[index].id) {
                return {StatusCode::invalid_argument, "participant port IDs must be unique"};
            }
        }
    }
    const auto status = participant_.initialize(time_.accepted_time);
    state_ = status.ok() ? SimulationState::initialized : SimulationState::failed;
    return status;
}

Status Simulation::prepare() {
    if (state_ != SimulationState::initialized) {
        return invalid_state("prepare requires an initialized simulation");
    }
    const auto status = participant_.prepare();
    state_ = status.ok() ? SimulationState::prepared : SimulationState::failed;
    return status;
}

Status Simulation::begin_window(double step) {
    if (state_ != SimulationState::prepared) {
        return invalid_state("begin_window requires a prepared simulation");
    }
    const auto constraint = participant_.time_constraint(time_.accepted_time);
    const auto valid = validate_time_step(constraint, step);
    if (!valid.ok()) {
        return valid;
    }
    const double endpoint = time_.accepted_time + step;
    if (!std::isfinite(endpoint) || endpoint <= time_.accepted_time) {
        return {StatusCode::invalid_argument, "step does not produce a finite advancing endpoint"};
    }
    if (time_.accepted_steps == std::numeric_limits<std::uint64_t>::max() ||
        time_.trial_attempts == std::numeric_limits<std::uint64_t>::max()) {
        return {StatusCode::resource_exhausted, "simulation step counter exhausted"};
    }
    const TimeInterval next_interval{time_.accepted_time, endpoint, step};
    const auto status = participant_.begin_window(next_interval);
    if (status.ok()) {
        interval_ = next_interval;
        state_ = SimulationState::window_open;
    }
    return status;
}

Status Simulation::reject_trial(const Status& cause) {
    const auto restored = restore_window();
    if (!restored.ok()) {
        return {restored.code(), cause.message() + "; rollback failed: " + restored.message()};
    }
    return cause;
}

Status Simulation::advance_trial() {
    if (state_ != SimulationState::window_open) {
        return invalid_state("advance_trial requires an open window");
    }
    ++time_.trial_attempts;
    const auto status = participant_.advance_trial(interval_);
    if (!status.ok())
        return reject_trial(status);
    state_ = SimulationState::trial_advanced;
    return {};
}

Status Simulation::commit_window() {
    if (state_ != SimulationState::trial_advanced) {
        return invalid_state("commit_window requires a successful trial");
    }
    auto status = participant_.prepare_commit(interval_);
    if (!status.ok()) {
        return reject_trial(status);
    }
    status = participant_.commit_window(interval_);
    if (!status.ok()) {
        return reject_trial(status);
    }
    time_.accepted_time = interval_.end;
    ++time_.accepted_steps;
    state_ = SimulationState::prepared;
    return {};
}

Status Simulation::restore_window() {
    if (state_ != SimulationState::window_open && state_ != SimulationState::trial_advanced) {
        return invalid_state("restore_window requires an open window");
    }
    const auto status = participant_.restore_window();
    state_ = status.ok() ? SimulationState::prepared : SimulationState::failed;
    return status;
}

Status Simulation::advance(double step) {
    auto status = begin_window(step);
    if (!status.ok()) {
        return status;
    }
    status = advance_trial();
    return status.ok() ? commit_window() : status;
}

Status Simulation::finalize() {
    if (state_ == SimulationState::finalized) {
        return invalid_state("simulation is already finalized");
    }
    if (state_ == SimulationState::window_open || state_ == SimulationState::trial_advanced) {
        const auto status = restore_window();
        if (!status.ok()) {
            return status;
        }
    }
    const auto status = participant_.finalize();
    state_ = status.ok() ? SimulationState::finalized : SimulationState::failed;
    return status;
}

} // namespace nexora
