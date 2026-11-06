#pragma once

#include <nexora/participant/ParticipantIF.hpp>

#include <cstdint>

namespace nexora {

enum class SimulationState {
    created,
    initialized,
    prepared,
    window_open,
    trial_advanced,
    failed,
    finalized
};

struct TimeState {
    double accepted_time = 0.0;
    std::uint64_t accepted_steps = 0;
    std::uint64_t trial_attempts = 0;
};

// Synchronous orchestration of exactly one participant. The caller owns the
// participant and must keep it alive and exclusively controlled by Simulation.
// Destruction does not call participant methods; finalize explicitly.
class Simulation {
  public:
    explicit Simulation(ParticipantIF& participant, double initial_time = 0.0) noexcept;
    Simulation(const Simulation&) = delete;
    Simulation& operator=(const Simulation&) = delete;
    Simulation(Simulation&&) = delete;
    Simulation& operator=(Simulation&&) = delete;

    Status initialize();
    Status prepare();
    Status begin_window(double step);
    Status advance_trial();
    Status commit_window();
    Status restore_window();

    // Begin, trial and commit one requested step. Trial/commit errors restore
    // the accepted checkpoint automatically; the caller chooses a legal retry.
    Status advance(double step);
    Status finalize();

    SimulationState state() const noexcept {
        return state_;
    }
    const TimeState& time() const noexcept {
        return time_;
    }

  private:
    Status reject_trial(const Status& cause);

    ParticipantIF& participant_;
    SimulationState state_ = SimulationState::created;
    TimeState time_;
    TimeInterval interval_;
};

} // namespace nexora
