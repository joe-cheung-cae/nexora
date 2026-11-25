#pragma once

#include <nexora/participant/ParticipantIF.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace nexora {

struct TimeState {
    double accepted_time = 0.0;
    std::uint64_t accepted_steps = 0;
};

/** @brief Synchronously advance one participant.
 *
 * The caller owns the participant and keeps it alive under this Simulation's
 * exclusive control. For a Kokkos-backed participant, keep the application's
 * `Kokkos::ScopeGuard` active through initialization, advancement, and cleanup,
 * and until the participant and any retained Kokkos-backed results are released.
 *
 * Call @ref initialize once, advance with `run()`, and call @ref finalize
 * explicitly while the runtime is active. Callbacks may observe the accepted
 * time via @ref time. Reentrant lifecycle calls return `StatusCode::invalid_state`.
 */
class Simulation {
  public:
    /** @brief Set the participant and starting clock for a simulation.
     * @param participant The caller-owned participant to advance.
     * @param initial_time The initial absolute time. @ref initialize requires a finite value.
     */
    explicit Simulation(ParticipantIF& participant, double initial_time = 0.0) noexcept;
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;
    Simulation(Simulation&&) = delete;
    Simulation& operator=(Simulation&&) = delete;

    /// @brief Validate the starting clock and ports, then initialize the participant.
    Status initialize();

    /** @brief Run a planned sequence of requested durations in order.
     *
     * Each successful step advances the accepted clock and step count once.
     * If a step fails, `run()` returns its status and the caller can still read
     * the last accepted participant state and clock. Call @ref finalize
     * explicitly after failure.
     *
     * @param steps Requested durations to pass to the participant.
     * @return Success when the sequence completes, or the first failure status.
     */
    Status run(std::span<const double> steps);

    /** @brief Run a sequence with one requested duration repeated.
     *
     * As with a planned sequence, a failed step leaves the last accepted state
     * and clock readable and requires explicit @ref finalize.
     *
     * @param step Requested duration for each step.
     * @param count Number of steps to request.
     * @return Success when the sequence completes, or the first failure status.
     */
    Status run(double step, std::size_t count = 1);

    /** @brief Explicitly finalize the participant and return its cleanup status.
     *
     * Call after the last successful run or after an initialization or run
     * failure. The caller is responsible for cleanup before destruction.
     */
    Status finalize();

    /// @brief Observe the last accepted clock and step count.
    const TimeState& time() const noexcept {
        return time_;
    }

  private:
    enum class Phase { created, ready, failed, finalized };
    Status run_schedule(std::span<const double> steps, std::size_t repetitions);

    /// The caller owns this participant and keeps it alive through explicit cleanup.
    ParticipantIF& participant_;
    Phase phase_ = Phase::created;
    bool operation_active_{};
    /// Stores the accepted clock and step count by value.
    TimeState time_;
};

} // namespace nexora
