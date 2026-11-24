#pragma once

#include <nexora/core/Status.hpp>
#include <nexora/participant/Port.hpp>
#include <nexora/participant/TimeConstraint.hpp>

#include <vector>

namespace nexora {

struct ParticipantCapabilities {
    bool rollback = false;
};

// All operations complete synchronously and report operational failure through
// Status. Implementations must not throw from these lifecycle methods.
// This minimal API does not claim asynchronous, distributed or coupling support.
class ParticipantIF {
  public:
    virtual ~ParticipantIF() = default;
    virtual ParticipantId id() const noexcept = 0;
    virtual ParticipantCapabilities capabilities() const = 0;
    virtual std::vector<PortDescriptor> ports() const = 0;
    virtual TimeConstraint time_constraint(double accepted_time) const = 0;

    virtual Status initialize(double initial_time) = 0;
    virtual Status prepare() = 0;

    // Capture a complete accepted checkpoint. Failure leaves accepted state
    // unchanged and no open window. Only one checkpoint/window may be active.
    virtual Status begin_window(const TimeInterval& interval) = 0;
    // Success completes the full interval.duration(). Absolute endpoints are rounded
    // time labels, so subtracting them is not an equivalent duration calculation.
    virtual Status advance_trial(const TimeInterval& interval) = 0;

    // Complete all fallible precommit work here. No accepted output is emitted.
    virtual Status prepare_commit(const TimeInterval& interval) = 0;
    // On success publish the trial and release the checkpoint. On failure the
    // checkpoint must remain restorable; do not emit irreversible output.
    virtual Status commit_window(const TimeInterval& interval) = 0;
    // Restore ALL state affecting replay, then discard the trial/checkpoint.
    // It must also be callable after a partially failed trial or commit.
    virtual Status restore_window() = 0;

    // Cleanup must be safe after partial initialization and failed restoration.
    virtual Status finalize() = 0;
};

} // namespace nexora
