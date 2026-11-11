#include <methods/reference/State.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace nexora::reference {
namespace {
Status invalid_state(const char* message) {
    return {StatusCode::invalid_state, message};
}
Status invalid_argument(const char* message) {
    return {StatusCode::invalid_argument, message};
}
} // namespace

State::State(Config config) : config_(std::move(config)) {}

Status State::initialize(double time) {
    if (phase_ != Phase::created) {
        return invalid_state("reference state already initialized");
    }
    if (!Kokkos::is_initialized() || Kokkos::is_finalized()) {
        return invalid_state("Kokkos must be initialized by the application");
    }
    if (!std::isfinite(time) || !std::isfinite(config_.lambda) || config_.lambda <= 0 ||
        config_.ids.empty() || config_.ids.size() != config_.x.size() ||
        config_.ids.size() != config_.q.size()) {
        return invalid_argument("invalid reference initial state or lambda");
    }
    if (config_.ids.size() > std::numeric_limits<LocalIndex>::max()) {
        return {StatusCode::resource_exhausted, "reference entity count exceeds local index capacity"};
    }
    auto sorted = config_.ids;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        return invalid_argument("reference stable IDs must be unique");
    }
    for (std::size_t i = 0; i < config_.ids.size(); ++i) {
        if (!std::isfinite(config_.x[i]) || !std::isfinite(config_.q[i])) {
            return invalid_argument("reference initial values must be finite");
        }
    }
    const auto n = config_.ids.size();
    Kokkos::View<std::uint64_t*> ids("reference.ids", n);
    Values x("reference.accepted.x", n), q("reference.accepted.q", n);
    auto hi = Kokkos::create_mirror(ids);
    auto hx = Kokkos::create_mirror(x);
    auto hq = Kokkos::create_mirror(q);
    for (std::size_t i = 0; i < n; ++i) {
        hi(i) = config_.ids[i];
        hx(i) = config_.x[i];
        hq(i) = config_.q[i];
    }
    Kokkos::deep_copy(ids, hi);
    Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(q, hq);
    ids_ = ids;
    accepted_x_ = x;
    accepted_q_ = q;
    time_ = time;
    std::vector<std::uint64_t>().swap(config_.ids);
    std::vector<real>().swap(config_.x);
    std::vector<real>().swap(config_.q);
    phase_ = Phase::initialized;
    return {};
}

Status State::prepare() {
    if (phase_ != Phase::initialized) {
        return invalid_state("prepare requires initialized reference state");
    }
    phase_ = Phase::ready;
    return {};
}

bool State::matches(const TimeInterval& i) const noexcept {
    return i.begin == interval_.begin && i.end == interval_.end &&
           i.duration() == interval_.duration();
}

FieldStamp State::accepted_stamp() const noexcept {
    return Snapshot::accepted_stamp(StateVersion{version_}, time_);
}

FieldStamp State::trial_stamp() const noexcept {
    FieldStamp stamp;
    const auto window = version_ + 1;
    stamp.accepted_version = StateVersion{version_};
    stamp.trial = TrialEpoch{window, current_attempt_, 0, 0};
    stamp.time = TimeStamp{interval_.end, window, 0, 0};
    stamp.valid_from = stamp.valid_until = interval_.end;
    return stamp;
}

FieldStamp State::next_accepted_stamp() const noexcept {
    return Snapshot::accepted_stamp(StateVersion{version_ + 1}, interval_.end);
}

Status State::begin(const TimeInterval& i) {
    if (phase_ != Phase::ready) {
        return invalid_state("begin requires prepared reference state");
    }
    if (!std::isfinite(i.begin) || !std::isfinite(i.end) || !std::isfinite(i.duration()) ||
        i.begin != time_ || i.duration() <= 0 || i.end <= i.begin ||
        i.begin + i.duration() != i.end) {
        return invalid_argument("invalid reference time interval");
    }
    const double requested_factor = i.duration() * static_cast<double>(config_.lambda);
    if (!std::isfinite(requested_factor) || requested_factor <= 0 || requested_factor > 1) {
        return invalid_argument("reference step requires 0 < dt * lambda <= 1");
    }
    if (i.duration() > static_cast<double>(std::numeric_limits<real>::max())) {
        return invalid_argument("reference duration exceeds real range");
    }
    const real dt = static_cast<real>(i.duration());
    if (!std::isfinite(dt) || dt <= real{0}) {
        return invalid_argument("reference duration is not positive and finite in real precision");
    }
    const real factor = dt * config_.lambda;
    if (!std::isfinite(factor) || factor <= real{0} || factor > real{1}) {
        return invalid_argument("reference real step requires 0 < dt * lambda <= 1");
    }
    if (next_attempt_ == std::numeric_limits<std::uint64_t>::max()) {
        return {StatusCode::resource_exhausted, "reference attempt counter exhausted"};
    }
    if (version_ == std::numeric_limits<std::uint64_t>::max()) {
        return {StatusCode::resource_exhausted, "reference state version exhausted"};
    }
    Values x("reference.trial.x", accepted_x_.extent(0));
    Values q("reference.trial.q", accepted_q_.extent(0));
    trial_x_ = x;
    trial_q_ = q;
    interval_ = i;
    trial_dt_ = dt;
    current_attempt_ = ++next_attempt_;
    phase_ = Phase::open;
    return {};
}

Status State::check_finite(const ConstValues& values) {
    auto host = Kokkos::create_mirror(values);
    Kokkos::deep_copy(host, values);
    for (std::size_t j = 0; j < values.extent(0); ++j) {
        if (!std::isfinite(host(j))) {
            return {StatusCode::numerical_failure, "reference trial produced nonfinite values"};
        }
    }
    return {};
}

Status State::advance(const TimeInterval& i, const Kokkos::Serial& exec, const ConstValues& x,
                      const ConstValues& q, const Values& tx, const Values& tq) {
    if (phase_ != Phase::open) {
        return invalid_state("advance requires an open reference window");
    }
    if (!matches(i)) {
        return invalid_argument("reference interval does not match open window");
    }
    // Mark the trial noncommittable before any fallible launch or completion.
    phase_ = Phase::failed_trial;
    // Use exactly the real duration validated when this window was opened.
    const real dt = trial_dt_, lambda = config_.lambda;
    using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::uint64_t>>;
    Kokkos::parallel_for(
        "reference.relaxation", Policy(exec, 0, x.extent(0)), KOKKOS_LAMBDA(const std::uint64_t j) {
            const real d = dt * lambda * x(j);
            tx(j) = x(j) - d;
            tq(j) = q(j) + d;
        });
    exec.fence("reference trial complete");
    if (const auto status = check_finite(tx); !status.ok()) {
        return status;
    }
    if (const auto status = check_finite(tq); !status.ok()) {
        return status;
    }
    phase_ = Phase::advanced;
    return {};
}

Status State::check_commit(const TimeInterval& i) const {
    if (phase_ != Phase::advanced) {
        return invalid_state("prepare_commit requires an advanced reference trial");
    }
    if (!matches(i)) {
        return invalid_argument("reference interval does not match open window");
    }
    return {};
}

Status State::commit(const TimeInterval& i) {
    if (phase_ != Phase::commit_ready) {
        return invalid_state("commit requires prepared reference trial");
    }
    if (!matches(i)) {
        return invalid_argument("reference interval does not match open window");
    }
    accepted_x_ = trial_x_;
    accepted_q_ = trial_q_;
    trial_x_ = {};
    trial_q_ = {};
    time_ = i.end;
    ++version_;
    accepted_attempt_ = current_attempt_;
    phase_ = Phase::ready;
    return {};
}

Status State::restore() {
    if (phase_ != Phase::open && phase_ != Phase::advanced && phase_ != Phase::failed_trial &&
        phase_ != Phase::commit_ready) {
        return invalid_state("restore requires an open reference window");
    }
    trial_x_ = {};
    trial_q_ = {};
    current_attempt_ = 0;
    phase_ = Phase::ready;
    return {};
}

Status State::finalize() {
    if (phase_ == Phase::finalized) {
        return invalid_state("reference state already finalized");
    }
    trial_x_ = {};
    trial_q_ = {};
    accepted_x_ = {};
    accepted_q_ = {};
    ids_ = {};
    phase_ = Phase::finalized;
    return {};
}

FieldDescriptor State::field_descriptor(TrialField field, bool accepted, FieldAccess access) {
    const bool is_x = field == TrialField::x;
    FieldDescriptor descriptor;
    descriptor.id = FieldId{is_x ? std::uint64_t{1} : std::uint64_t{2}};
    descriptor.name = is_x ? "x" : "q";
    descriptor.association = Association::node;
    descriptor.entity_set = "reference_samples";
    descriptor.units = "1";
    descriptor.scalar_type = real_scalar_type;
    descriptor.components = 1;
    descriptor.measure = MeasureSemantics::point_value;
    descriptor.access = access;
    descriptor.time_level = accepted ? "accepted" : "trial";
    return descriptor;
}

Result<Snapshot> State::snapshot() const {
    if (phase_ == Phase::created || phase_ == Phase::finalized) {
        return invalid_state("reference snapshot requires initialized state");
    }
    Snapshot out;
    out.ids = ids_;
    out.version = StateVersion{version_};
    out.time = time_;
    out.attempt = accepted_attempt_;
    out.x_descriptor_ = field_descriptor(TrialField::x, true, FieldAccess::read);
    out.q_descriptor_ = field_descriptor(TrialField::q, true, FieldAccess::read);
    return out;
}
} // namespace nexora::reference
