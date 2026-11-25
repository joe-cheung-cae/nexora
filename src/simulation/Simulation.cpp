#include <nexora/simulation/Simulation.hpp>

#include <cmath>
#include <limits>

namespace nexora {
namespace {

Status invalid_state(const char* message) {
    return {StatusCode::invalid_state, message};
}

class ActiveOperation {
  public:
    explicit ActiveOperation(bool& active) noexcept : active_(active) { active_ = true; }
    ~ActiveOperation() { active_ = false; }

    ActiveOperation(const ActiveOperation&) = delete;
    ActiveOperation& operator=(const ActiveOperation&) = delete;

  private:
    bool& active_;
};

Status reject_reentry() {
    return invalid_state("simulation operation is already active");
}

} // namespace

Simulation::Simulation(ParticipantIF& participant, double initial_time) noexcept
    : participant_(participant) {
    time_.accepted_time = initial_time;
}

Status Simulation::initialize() {
    if (operation_active_)
        return reject_reentry();
    if (phase_ != Phase::created)
        return invalid_state("simulation can only initialize once");

    const ActiveOperation operation(operation_active_);
    phase_ = Phase::failed;
    if (!std::isfinite(time_.accepted_time))
        return {StatusCode::invalid_argument, "initial time must be finite"};
    const auto ports = participant_.ports();
    for (std::size_t index = 0; index < ports.size(); ++index) {
        const auto status = ports[index].validate();
        if (!status.ok())
            return status;
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (ports[previous].id == ports[index].id)
                return {StatusCode::invalid_argument, "participant port IDs must be unique"};
        }
    }
    const auto status = participant_.initialize(time_.accepted_time);
    if (status.ok())
        phase_ = Phase::ready;
    return status;
}

Status Simulation::run(std::span<const double> steps) {
    return run_schedule(steps, 1);
}

Status Simulation::run(double step, std::size_t count) {
    return run_schedule(std::span<const double>{&step, 1}, count);
}

Status Simulation::run_schedule(std::span<const double> steps, std::size_t repetitions) {
    if (operation_active_)
        return reject_reentry();
    if (phase_ != Phase::ready)
        return invalid_state("run requires an initialized simulation without a prior failure");

    const ActiveOperation operation(operation_active_);
    phase_ = Phase::failed;
    for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
        for (double step : steps) {
            if (time_.accepted_steps == std::numeric_limits<std::uint64_t>::max())
                return {StatusCode::resource_exhausted, "simulation step counter exhausted"};
            const TimeInterval interval{time_.accepted_time, time_.accepted_time + step, step};
            const auto status = participant_.step(interval);
            if (!status.ok())
                return status;
            time_.accepted_time = interval.end;
            ++time_.accepted_steps;
        }
    }
    phase_ = Phase::ready;
    return {};
}

Status Simulation::finalize() {
    if (operation_active_)
        return reject_reentry();
    if (phase_ == Phase::finalized)
        return invalid_state("simulation is already finalized");

    const ActiveOperation operation(operation_active_);
    const auto status = participant_.finalize();
    phase_ = status.ok() ? Phase::finalized : Phase::failed;
    return status;
}

} // namespace nexora
