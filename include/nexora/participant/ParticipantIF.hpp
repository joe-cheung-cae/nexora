#pragma once

#include <nexora/core/Status.hpp>
#include <nexora/participant/Port.hpp>
#include <nexora/participant/TimeConstraint.hpp>

#include <cmath>
#include <vector>

namespace nexora {

/** @brief Synchronous lifecycle and time-step interface for a simulation participant.
 *
 * Lifecycle methods complete their work before returning and report operational
 * failures through @ref Status. Implementations must keep exceptions inside the
 * lifecycle methods and return a failure status.
 */
class ParticipantIF {
  public:
    virtual ~ParticipantIF() = default;
    virtual ParticipantId id() const noexcept = 0;
    virtual std::vector<PortDescriptor> ports() const = 0;
    /** @brief Report the time constraint of the current accepted state.
     *
     * Implementations must keep this query free of user callbacks.
     */
    virtual TimeConstraint time_constraint() const = 0;

    /** @brief Establish the initial accepted state and its time constraint.
     * @param initial_time The absolute starting time.
     */
    virtual Status initialize(double initial_time) = 0;

    /** @brief Admit a requested step before dispatching to @ref do_step.
     *
     * This method checks that the requested duration is finite and positive and
     * satisfies the current accepted time constraint, including its bounds and
     * any fixed-quantum requirement. It then verifies that the rounded endpoint
     * is finite, advances from the beginning, and equals the beginning plus the
     * requested duration.
     *
     * @param interval Absolute endpoints and authoritative @ref TimeInterval::duration.
     * @return The admission failure status or the result of @ref do_step.
     */
    Status step(const TimeInterval& interval) {
        const auto status = validate_time_step(time_constraint(), interval.duration());
        if (!status.ok())
            return status;
        if (!std::isfinite(interval.end) || !(interval.end > interval.begin) ||
            interval.end != interval.begin + interval.duration()) {
            return {StatusCode::invalid_argument, "step does not produce a finite advancing endpoint"};
        }
        return do_step(interval);
    }

    /** @brief Complete outstanding work and release participant-owned resources.
     *
     * Cleanup must be safe after partial initialization and a failed step.
     */
    virtual Status finalize() = 0;

  protected:
    /** @brief Implement the synchronous advance for an admitted interval.
     *
     * On success, publish the complete requested step and the next state's time
     * constraint together. On failure, finish outstanding work, release temporary
     * state and its grants, and preserve the previous accepted results and constraint.
     *
     * @param interval The interval admitted by @ref step. Use @ref TimeInterval::duration
     * for the integration duration.
     * @return Success after publication, or a failure status with the accepted state intact.
     */
    virtual Status do_step(const TimeInterval& interval) = 0;

  private:
    friend class Simulation;
    // Simulation has checked the clock. Existing participants can keep their
    // ordinary initialize override; built-in implementations share its body.
    virtual Status initialize_at_finite_time(double time) { return initialize(time); }
};

} // namespace nexora
