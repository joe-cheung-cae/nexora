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

// Synchronous orchestration of exactly one participant. The caller owns the
// participant and must keep it alive and exclusively controlled by Simulation.
// Callbacks within a Simulation operation may observe the accepted time.
// Reentrant lifecycle operations on that Simulation return invalid_state.
// Destruction does not call participant methods; finalize explicitly.
class Simulation {
  public:
    explicit Simulation(ParticipantIF& participant, double initial_time = 0.0) noexcept;
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;
    Simulation(Simulation&&) = delete;
    Simulation& operator=(Simulation&&) = delete;

    Status initialize();
    // Run a planned sequence of requested durations, or repeat one duration.
    // After a failure, the caller can read accepted results and finalize.
    Status run(std::span<const double> steps);
    Status run(double step, std::size_t count = 1);
    Status finalize();

    const TimeState& time() const noexcept {
        return time_;
    }

  private:
    enum class Phase { created, ready, failed, finalized };
    Status run_schedule(std::span<const double> steps, std::size_t repetitions);

    ParticipantIF& participant_;
    Phase phase_ = Phase::created;
    bool operation_active_{};
    TimeState time_;
};

} // namespace nexora
