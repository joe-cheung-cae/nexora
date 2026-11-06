#pragma once

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <nexora/core/Result.hpp>
#include <nexora/core/Time.hpp>
#include <nexora/data/FieldDescriptor.hpp>
#include <vector>

namespace nexora::reference {

struct Config {
    std::vector<std::uint64_t> ids{0};
    std::vector<double> x{1.0};
    std::vector<double> q{0.0};
    double lambda = 1.0;
};

// Managed read-only views retain their allocations. Destroy all snapshots before
// application-owned Kokkos finalization, even if the participant is gone.
class Snapshot {
  public:
    Kokkos::View<const std::uint64_t*> ids;
    Kokkos::View<const double*> x, q;
    StateVersion version{};
    double time = 0.0;
    std::uint64_t attempt = 0;
    const FieldDescriptor& x_descriptor() const noexcept {
        return x_descriptor_;
    }
    const FieldDescriptor& q_descriptor() const noexcept {
        return q_descriptor_;
    }

  private:
    friend class State;
    FieldDescriptor x_descriptor_, q_descriptor_;
};

class State {
  public:
    explicit State(Config config = {});
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = delete;
    State& operator=(State&&) = delete;
    Status initialize(double time);
    Status prepare();
    Status begin(const TimeInterval& interval);
    Status advance(const TimeInterval& interval);
    Status prepare_commit(const TimeInterval& interval);
    Status commit(const TimeInterval& interval);
    Status restore();
    Status finalize();
    Result<Snapshot> snapshot() const;
    double lambda() const noexcept {
        return config_.lambda;
    }

  private:
    enum class Phase {
        created,
        initialized,
        ready,
        open,
        advanced,
        failed_trial,
        commit_ready,
        finalized
    };
    bool matches(const TimeInterval& interval) const noexcept;
    Config config_;
    Phase phase_ = Phase::created;
    Kokkos::View<const std::uint64_t*> ids_;
    Kokkos::View<const double*> accepted_x_, accepted_q_;
    Kokkos::View<double*> trial_x_, trial_q_;
    TimeInterval interval_{};
    double time_ = 0.0;
    std::uint64_t version_ = 0, next_attempt_ = 0, accepted_attempt_ = 0, current_attempt_ = 0;
};
} // namespace nexora::reference
