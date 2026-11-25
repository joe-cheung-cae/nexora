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
#include <vector>

namespace nexora::reference {

enum class Field { x, q };
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

    enum class Phase { created, initialized, finalized };
    Result<AllocationSize> preflight_initialize(double time) const;
    Result<std::size_t> preflight_step(const TimeInterval& interval) const;
    Status initialize(double time, const Kokkos::Serial& exec);
    void cancel_initialization() noexcept;
    void finish_initialization() noexcept;
    static Status evaluate(const Kokkos::Serial& exec, real dt, real lambda,
                           const ConstValues& x, const ConstValues& q,
                           const Values& next_x, const Values& next_q,
                           const History& history, std::uint64_t sequence);
    Status finalize();
    Result<Snapshot> snapshot() const;
    static FieldDescriptor field_descriptor(Field field, FieldAccess access);
    static FieldDescriptor history_descriptor(FieldAccess access);
    FieldStamp accepted_stamp() const noexcept;
    Config config_;
    Phase phase_ = Phase::created;
    ConstEntityIds ids_;
    ConstValues accepted_x_, accepted_q_;
    ConstHistory accepted_history_;
    Control accepted_control_;
    std::uint64_t accepted_diagnostic_ = 0;
    double time_ = 0.0;
    std::uint64_t version_ = 0;
};
} // namespace nexora::reference
