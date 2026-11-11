#pragma once

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <nexora/core/Real.hpp>
#include <nexora/core/Result.hpp>
#include <nexora/core/Time.hpp>
#include <nexora/data/FieldBinding.hpp>
#include <nexora/data/FieldDescriptor.hpp>
#include <nexora/data/FieldStamp.hpp>
#include <optional>
#include <vector>

namespace nexora::reference {

enum class TrialField { x, q };
using Values = Kokkos::View<real*, Kokkos::HostSpace>;
using ConstValues = Values::const_type;
using EntityIds = Kokkos::View<std::uint64_t*, Kokkos::HostSpace>;
using ConstEntityIds = EntityIds::const_type;
using History = Kokkos::View<std::uint64_t*, Kokkos::HostSpace>;
using ConstHistory = History::const_type;

// Replay fixture only: neither history nor this counter changes x/q physics.
struct Control {
    std::uint64_t sequence = 0;
    friend bool operator==(const Control&, const Control&) = default;
};

struct DiagnosticKey {
    FieldStamp stamp;
    friend bool operator==(const DiagnosticKey&, const DiagnosticKey&) = default;
};

struct Config {
    std::vector<std::uint64_t> ids{0};
    std::vector<real> x{real{1}};
    std::vector<real> q{real{0}};
    real lambda = real{1};
    std::uint64_t history_seed = 0;
    Control control_seed{};
};

// Pure input checks shared by application preflight and the allocating lifecycle.
// They neither initialize Kokkos nor allocate numerical storage.
struct AllocationSize {
    std::size_t ids_bytes = 0;
    std::size_t bank_bytes = 0;
};
Result<AllocationSize> validate_config(const Config& config, double time);
Status validate_interval(const TimeInterval& interval, real lambda);

// Copies share the read grants on accepted values. Destroy all snapshots before
// application-owned Kokkos finalization, even if the participant is gone.
// Keep a Snapshot alive while accessing any extracted native View: View copies
// alone do not preserve the controlled lease or budget reservation.
class Snapshot {
    struct Grants;
    // Declared before native Views, so their grants and reservations release last.
    std::shared_ptr<const Grants> grants_;

  public:
    Snapshot() = default;
    Snapshot(const Snapshot&) = default;
    Snapshot(Snapshot&&) noexcept = default;
    Snapshot& operator=(Snapshot other) noexcept;
    ConstEntityIds ids;
    ConstValues x, q;
    ConstHistory history;
    Control control;
    std::uint64_t diagnostic = 0;
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
        // Reservations outlive their typed bindings, including during assignment.
        std::shared_ptr<const void> ids_budget, bank_budget, snapshot_budget;
        ReadField<ConstValues> x, q;
        ReadField<ConstHistory> history;
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
        failed,
        finalized
    };

    Result<AllocationSize> preflight_initialize(double time) const;
    Result<std::size_t> preflight_begin(const TimeInterval& interval) const;
    // Participant must preflight and reserve the returned payload size before
    // these private allocation stages, without intervening state mutation.
    Status initialize(double time, const Kokkos::Serial& exec);
    void cancel_initialization() noexcept;
    void finish_initialization() noexcept;
    Status prepare();
    Status begin(const TimeInterval& interval, const Kokkos::Serial& exec,
                 const ConstHistory& accepted_history);
    Status advance(const TimeInterval& interval, const Kokkos::Serial& exec, const ConstValues& x,
                   const ConstValues& q, const Values& tx, const Values& tq,
                   const History& history, const FieldStamp& history_stamp);
    Status check_commit(const TimeInterval& interval, const FieldStamp& history_stamp) const;
    Status commit(const TimeInterval& interval);
    Status restore();
    Status finalize();
    Result<Snapshot> snapshot() const;
    Result<DiagnosticKey> trial_diagnostic_key() const;
    Result<std::uint64_t> trial_diagnostic(const DiagnosticKey& key) const;
    static Status check_finite(const Kokkos::Serial& exec, const ConstValues& values);
    static Status check_finite(const Kokkos::Serial& exec, const ConstValues& x,
                               const ConstValues& q);
    static FieldDescriptor field_descriptor(TrialField field, bool accepted, FieldAccess access);
    static FieldDescriptor history_descriptor(bool accepted, FieldAccess access);
    FieldStamp accepted_stamp() const noexcept;
    FieldStamp trial_stamp() const noexcept;
    FieldStamp next_accepted_stamp() const noexcept;
    bool matches(const TimeInterval& interval) const noexcept;
    Config config_;
    Phase phase_ = Phase::created;
    ConstEntityIds ids_;
    ConstValues accepted_x_, accepted_q_;
    Values trial_x_, trial_q_;
    ConstHistory accepted_history_;
    History trial_history_;
    Control accepted_control_, trial_control_;
    std::uint64_t accepted_diagnostic_ = 0;
    struct DiagnosticCache {
        // Only history/control contribute; x/q receive does not invalidate it.
        // The authority stamp carries layout, accepted version, attempt and
        // storage incarnation. Rebuilt after advance; discarded with its trial.
        DiagnosticKey key;
        std::uint64_t value = 0;
    };
    std::optional<DiagnosticCache> diagnostic_cache_;
    TimeInterval interval_{};
    real trial_dt_ = real{0};
    double time_ = 0.0;
    std::uint64_t version_ = 0, next_attempt_ = 0, accepted_attempt_ = 0, current_attempt_ = 0;
};
} // namespace nexora::reference
