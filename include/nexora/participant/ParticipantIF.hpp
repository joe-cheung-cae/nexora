#pragma once

#include <nexora/core/Status.hpp>
#include <nexora/participant/Port.hpp>
#include <nexora/participant/TimeConstraint.hpp>

#include <cmath>
#include <vector>

namespace nexora {

// All operations complete synchronously and report operational failure through
// Status. Implementations must not throw from these lifecycle methods.
// This minimal API does not claim asynchronous, distributed or coupling support.
class ParticipantIF {
  public:
    virtual ~ParticipantIF() = default;
    virtual ParticipantId id() const noexcept = 0;
    virtual std::vector<PortDescriptor> ports() const = 0;
    // Describe the current accepted state. The query must not invoke user callbacks.
    virtual TimeConstraint time_constraint() const = 0;

    virtual Status initialize(double initial_time) = 0;

    // Check the accepted constraint once before checking the rounded endpoint.
    Status step(const TimeInterval& interval) {
        const auto status = validate_time_step(time_constraint(), interval.duration());
        if (!status.ok())
            return status;
        if (!std::isfinite(interval.begin) || !std::isfinite(interval.end) ||
            interval.end <= interval.begin || interval.end != interval.begin + interval.duration()) {
            return {StatusCode::invalid_argument, "step does not produce a finite advancing endpoint"};
        }
        return do_step(interval);
    }

    // Cleanup must be safe after partial initialization and a failed step.
    virtual Status finalize() = 0;

  protected:
    // Success publishes the full requested step and the next state's constraint.
    // Failure releases temporary state and preserves all accepted results.
    virtual Status do_step(const TimeInterval& interval) = 0;
};

} // namespace nexora
