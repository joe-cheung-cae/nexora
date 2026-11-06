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
    Kokkos::View<double*> x("reference.accepted.x", n), q("reference.accepted.q", n);
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
    std::vector<double>().swap(config_.x);
    std::vector<double>().swap(config_.q);
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

Status State::begin(const TimeInterval& i) {
    if (phase_ != Phase::ready) {
        return invalid_state("begin requires prepared reference state");
    }
    if (!std::isfinite(i.begin) || !std::isfinite(i.end) || !std::isfinite(i.duration()) ||
        i.begin != time_ || i.duration() <= 0 || i.end <= i.begin ||
        i.begin + i.duration() != i.end) {
        return invalid_argument("invalid reference time interval");
    }
    const double factor = i.duration() * config_.lambda;
    if (!std::isfinite(factor) || factor <= 0 || factor > 1) {
        return invalid_argument("reference step requires 0 < dt * lambda <= 1");
    }
    if (next_attempt_ == std::numeric_limits<std::uint64_t>::max()) {
        return {StatusCode::resource_exhausted, "reference attempt counter exhausted"};
    }
    Kokkos::View<double*> x("reference.trial.x", accepted_x_.extent(0));
    Kokkos::View<double*> q("reference.trial.q", accepted_q_.extent(0));
    trial_x_ = x;
    trial_q_ = q;
    interval_ = i;
    current_attempt_ = ++next_attempt_;
    phase_ = Phase::open;
    return {};
}

Status State::advance(const TimeInterval& i) {
    if (phase_ != Phase::open) {
        return invalid_state("advance requires an open reference window");
    }
    if (!matches(i)) {
        return invalid_argument("reference interval does not match open window");
    }
    // Mark the trial noncommittable before any fallible launch or completion.
    phase_ = Phase::failed_trial;
    auto x = accepted_x_;
    auto q = accepted_q_;
    auto tx = trial_x_;
    auto tq = trial_q_;
    const double dt = i.duration(), lambda = config_.lambda;
    Kokkos::Serial exec;
    using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::uint64_t>>;
    Kokkos::parallel_for(
        "reference.relaxation", Policy(exec, 0, x.extent(0)), KOKKOS_LAMBDA(const std::uint64_t j) {
            const double d = dt * lambda * x(j);
            tx(j) = x(j) - d;
            tq(j) = q(j) + d;
        });
    exec.fence("reference trial complete");
    auto hx = Kokkos::create_mirror(trial_x_);
    auto hq = Kokkos::create_mirror(trial_q_);
    Kokkos::deep_copy(hx, trial_x_);
    Kokkos::deep_copy(hq, trial_q_);
    for (std::size_t j = 0; j < trial_x_.extent(0); ++j) {
        if (!std::isfinite(hx(j)) || !std::isfinite(hq(j))) {
            return {StatusCode::numerical_failure, "reference trial produced nonfinite values"};
        }
    }
    phase_ = Phase::advanced;
    return {};
}

Status State::prepare_commit(const TimeInterval& i) {
    if (phase_ != Phase::advanced) {
        return invalid_state("prepare_commit requires an advanced reference trial");
    }
    if (!matches(i)) {
        return invalid_argument("reference interval does not match open window");
    }
    if (version_ == std::numeric_limits<std::uint64_t>::max()) {
        return {StatusCode::resource_exhausted, "reference state version exhausted"};
    }
    phase_ = Phase::commit_ready;
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

Result<Snapshot> State::snapshot() const {
    if (phase_ == Phase::created || phase_ == Phase::finalized) {
        return invalid_state("reference snapshot requires initialized state");
    }
    Snapshot out;
    out.ids = ids_;
    out.x = accepted_x_;
    out.q = accepted_q_;
    out.version = StateVersion{version_};
    out.time = time_;
    out.attempt = accepted_attempt_;
    auto populate = [&](FieldDescriptor& d, const char* name, std::uint64_t id) {
        d.id = FieldId{id};
        d.name = name;
        d.association = Association::node;
        d.entity_set = "reference_samples";
        d.units = "1";
        d.scalar_type = ScalarType::float64;
        d.components = 1;
        d.measure = MeasureSemantics::point_value;
        d.access = FieldAccess::read;
        d.valid_from = time_;
        d.valid_until = time_;
        d.time_level = "accepted";
        d.state_version = out.version;
    };
    populate(out.x_descriptor_, "x", 1);
    populate(out.q_descriptor_, "q", 2);
    return out;
}
} // namespace nexora::reference
