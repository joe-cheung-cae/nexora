#pragma once

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <memory>
#include <nexora/core/Real.hpp>
#include <nexora/core/Result.hpp>
#include <nexora/core/Time.hpp>
#include <nexora/data/FieldBinding.hpp>
#include <nexora/data/FieldDescriptor.hpp>
#include <nexora/data/FieldStamp.hpp>
#include <vector>

namespace nexora::reference {

enum class TrialField { x, q };
using Values = Kokkos::View<real*>;
using ConstValues = Values::const_type;

struct Config {
    std::vector<std::uint64_t> ids{0};
    std::vector<real> x{real{1}};
    std::vector<real> q{real{0}};
    real lambda = real{1};
};

// Copies share the read grants on accepted values. Destroy all snapshots before
// application-owned Kokkos finalization, even if the participant is gone.
class Snapshot {
  public:
    Kokkos::View<const std::uint64_t*> ids;
    ConstValues x, q;
    StateVersion version{};
    double time = 0.0;
    std::uint64_t attempt = 0;
    FieldStamp stamp() const noexcept {
        return accepted_stamp(version, time);
    }
    const FieldDescriptor& x_descriptor() const noexcept {
        return x_descriptor_;
    }
    const FieldDescriptor& q_descriptor() const noexcept {
        return q_descriptor_;
    }

  private:
    friend class Participant;
    friend class State;
    struct Grants {
        ReadField<ConstValues> x, q;
    };
    static FieldStamp accepted_stamp(StateVersion version, double time) noexcept {
        FieldStamp stamp;
        stamp.accepted_version = version;
        stamp.time.physical_time = time;
        stamp.valid_from = stamp.valid_until = time;
        stamp.accepted = true;
        return stamp;
    }
    FieldDescriptor x_descriptor_, q_descriptor_;
    std::shared_ptr<const Grants> grants_;
};

class State {
  public:
    explicit State(Config config = {});
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = delete;
    State& operator=(State&&) = delete;
    real lambda() const noexcept {
        return config_.lambda;
    }

  private:
    friend class Participant;

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

    Status initialize(double time);
    Status prepare();
    Status begin(const TimeInterval& interval);
    Status advance(const TimeInterval& interval, const Kokkos::Serial& exec, const ConstValues& x,
                   const ConstValues& q, const Values& tx, const Values& tq);
    Status check_commit(const TimeInterval& interval) const;
    Status commit(const TimeInterval& interval);
    Status restore();
    Status finalize();
    Result<Snapshot> snapshot() const;
    static Status check_finite(const ConstValues& values);
    static FieldDescriptor field_descriptor(TrialField field, bool accepted, FieldAccess access);
    FieldStamp accepted_stamp() const noexcept;
    FieldStamp trial_stamp() const noexcept;
    FieldStamp next_accepted_stamp() const noexcept;
    bool matches(const TimeInterval& interval) const noexcept;
    Config config_;
    Phase phase_ = Phase::created;
    Kokkos::View<const std::uint64_t*> ids_;
    ConstValues accepted_x_, accepted_q_;
    Values trial_x_, trial_q_;
    TimeInterval interval_{};
    real trial_dt_ = real{0};
    double time_ = 0.0;
    std::uint64_t version_ = 0, next_attempt_ = 0, accepted_attempt_ = 0, current_attempt_ = 0;
};
} // namespace nexora::reference
