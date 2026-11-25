#include <methods/reference/State.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace nexora::reference {
namespace {
Status invalid_state(const char* message) {
    return {StatusCode::invalid_state, message};
}
Status invalid_argument(const char* message) {
    return {StatusCode::invalid_argument, message};
}

template <class Exec, class ReadView, class WriteView>
void relaxation(const Exec& exec, real dt, real lambda, const ReadView& x,
                const ReadView& q, const WriteView& tx, const WriteView& tq) {
    static_assert(Kokkos::is_view<ReadView>::value && Kokkos::is_view<WriteView>::value);
    static_assert(ReadView::rank() == 1 && WriteView::rank() == 1);
    static_assert(std::is_same_v<typename ReadView::value_type, const real> &&
                  std::is_same_v<typename WriteView::value_type, real>);
    static_assert(Kokkos::SpaceAccessibility<Exec, typename ReadView::memory_space>::accessible &&
                  Kokkos::SpaceAccessibility<Exec, typename WriteView::memory_space>::accessible);
    using Policy = Kokkos::RangePolicy<Exec, Kokkos::IndexType<std::uint64_t>>;
    Kokkos::parallel_for(
        "reference.relaxation", Policy(exec, 0, x.extent(0)), KOKKOS_LAMBDA(const std::uint64_t j) {
            const real d = dt * lambda * x(j);
            tx(j) = x(j) - d;
            tq(j) = q(j) + d;
        });
}

template <class Exec, class View>
auto diagnostic_view(const Exec& exec, const View& values) {
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                             typename View::memory_space>::accessible) {
        return values;
    } else {
        auto host = Kokkos::create_mirror(Kokkos::WithoutInitializing, values);
        Kokkos::deep_copy(exec, host, values);
        return host;
    }
}

template <class View>
bool all_finite(const View& values) {
    for (std::size_t i = 0; i < values.extent(0); ++i) {
        if (!std::isfinite(values(i)))
            return false;
    }
    return true;
}

template <class T>
using HostInput = Kokkos::View<const T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
} // namespace

State::State(Config config) : config_(std::move(config)) {}

Snapshot& Snapshot::operator=(Snapshot other) noexcept {
    using std::swap;
    swap(ids, other.ids);
    swap(x, other.x);
    swap(q, other.q);
    swap(history, other.history);
    swap(control, other.control);
    swap(diagnostic, other.diagnostic);
    swap(version, other.version);
    swap(time, other.time);
    swap(x_descriptor_, other.x_descriptor_);
    swap(q_descriptor_, other.q_descriptor_);
    swap(grants_, other.grants_);
    return *this;
}

Result<AllocationSize> validate_config(const Config& config, double time) {
    if (!std::isfinite(time) || !std::isfinite(config.lambda) || config.lambda <= 0 ||
        config.ids.empty() || config.ids.size() != config.x.size() ||
        config.ids.size() != config.q.size()) {
        return invalid_argument("invalid reference initial state or lambda");
    }
    if (config.ids.size() > std::numeric_limits<LocalIndex>::max()) {
        return Status{StatusCode::resource_exhausted, "reference entity count exceeds local index capacity"};
    }
    auto sorted = config.ids;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        return invalid_argument("reference stable IDs must be unique");
    }
    for (std::size_t i = 0; i < config.ids.size(); ++i) {
        if (!std::isfinite(config.x[i]) || !std::isfinite(config.q[i])) {
            return invalid_argument("reference initial values must be finite");
        }
    }
    const auto n = config.ids.size();
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    if (n > max / sizeof(std::uint64_t) ||
        n > (max - sizeof(std::uint64_t)) / (2 * sizeof(real)))
        return Status{StatusCode::resource_exhausted, "reference state byte count overflow"};
    return AllocationSize{n * sizeof(std::uint64_t),
                          2 * n * sizeof(real) + sizeof(std::uint64_t)};
}

Result<AllocationSize> State::preflight_initialize(double time) const {
    if (phase_ != Phase::created) {
        return invalid_state("reference state already initialized");
    }
    if (!Kokkos::is_initialized() || Kokkos::is_finalized()) {
        return invalid_state("Kokkos must be initialized by the application");
    }
    return validate_config(config_, time);
}

Status State::initialize(double time, const Kokkos::Serial& exec) {
    // Participant already validated input and reserved IDs plus bank payload.
    const auto n = config_.ids.size();
    EntityIds ids(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "reference.ids"), n);
    Values x(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "reference.accepted.x"), n);
    Values q(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "reference.accepted.q"), n);
    History history(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                                       "reference.accepted.history"), 1);
    Kokkos::deep_copy(exec, history, config_.history_seed);
    // The config retains all three host buffers until the instance completes the copies.
    Kokkos::deep_copy(exec, ids, HostInput<std::uint64_t>(config_.ids.data(), n));
    Kokkos::deep_copy(exec, x, HostInput<real>(config_.x.data(), n));
    Kokkos::deep_copy(exec, q, HostInput<real>(config_.q.data(), n));
    exec.fence("reference initial state ready");
    ids_ = ids;
    accepted_x_ = x;
    accepted_q_ = q;
    accepted_history_ = history;
    accepted_control_ = config_.control_seed;
    accepted_diagnostic_ = config_.history_seed + config_.control_seed.sequence;
    time_ = time;
    phase_ = Phase::initialized;
    return {};
}

void State::finish_initialization() noexcept {
    std::vector<std::uint64_t>().swap(config_.ids);
    std::vector<real>().swap(config_.x);
    std::vector<real>().swap(config_.q);
}

void State::cancel_initialization() noexcept {
    ids_ = {};
    accepted_x_ = {};
    accepted_q_ = {};
    accepted_history_ = {};
    accepted_control_ = {};
    accepted_diagnostic_ = 0;
    time_ = 0;
    phase_ = Phase::created;
}

FieldStamp State::accepted_stamp() const noexcept {
    return Snapshot::accepted_stamp(StateVersion{version_}, time_);
}

Status validate_interval(const TimeInterval& i, real lambda) {
    if (!std::isfinite(i.begin) || !std::isfinite(i.end) || !std::isfinite(i.duration()) ||
        i.duration() <= 0 || i.end <= i.begin ||
        i.begin + i.duration() != i.end) {
        return invalid_argument("invalid reference time interval");
    }
    const double requested_factor = i.duration() * static_cast<double>(lambda);
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
    const real factor = dt * lambda;
    if (!std::isfinite(factor) || factor <= real{0} || factor > real{1}) {
        return invalid_argument("reference real step requires 0 < dt * lambda <= 1");
    }
    return {};
}

Result<std::size_t> State::preflight_step(const TimeInterval& i) const {
    if (phase_ != Phase::initialized)
        return invalid_state("step requires initialized reference state");
    if (i.begin != time_)
        return invalid_argument("invalid reference time interval");
    if (auto status = validate_interval(i, config_.lambda); !status.ok())
        return status;
    if (version_ == std::numeric_limits<std::uint64_t>::max())
        return Status{StatusCode::resource_exhausted, "reference state version exhausted"};
    if (accepted_control_.sequence == std::numeric_limits<std::uint64_t>::max())
        return Status{StatusCode::resource_exhausted, "reference control sequence exhausted"};
    return 2 * accepted_x_.extent(0) * sizeof(real) + sizeof(std::uint64_t);
}

Status State::evaluate(const Kokkos::Serial& exec, real dt, real lambda,
                        const ConstValues& x, const ConstValues& q,
                        const Values& next_x, const Values& next_q,
                        const History& history, std::uint64_t sequence) {
    relaxation(exec, dt, lambda, x, q, next_x, next_q);
    // Unsigned modular arithmetic is intentional in this history fixture.
    Kokkos::parallel_for("reference.history", Kokkos::RangePolicy<Kokkos::Serial>(exec, 0, 1),
                         KOKKOS_LAMBDA(const int) { history(0) = 33 * history(0) + sequence; });
    const auto host_x = diagnostic_view(exec, next_x);
    const auto host_q = diagnostic_view(exec, next_q);
    exec.fence("reference step diagnostics ready");
    if (!all_finite(host_x) || !all_finite(host_q))
        return {StatusCode::numerical_failure, "reference step produced nonfinite values"};
    return {};
}

Status State::finalize() {
    if (phase_ == Phase::finalized)
        return invalid_state("reference state already finalized");
    accepted_x_ = {};
    accepted_q_ = {};
    accepted_history_ = {};
    ids_ = {};
    finish_initialization();
    phase_ = Phase::finalized;
    return {};
}

FieldDescriptor State::field_descriptor(Field field, FieldAccess access) {
    const bool is_x = field == Field::x;
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
    descriptor.time_level = access == FieldAccess::read ? "accepted" : "step";
    return descriptor;
}

FieldDescriptor State::history_descriptor(FieldAccess access) {
    auto descriptor = field_descriptor(Field::x, access);
    descriptor.id = FieldId{4};
    descriptor.name = "replay_history";
    descriptor.entity_set = "reference_replay_history";
    descriptor.scalar_type = ScalarType::uint64;
    return descriptor;
}

Result<Snapshot> State::snapshot() const {
    if (phase_ == Phase::created || phase_ == Phase::finalized) {
        return invalid_state("reference snapshot requires initialized state");
    }
    Snapshot out;
    out.ids = ids_;
    out.history = accepted_history_;
    out.control = accepted_control_;
    out.diagnostic = accepted_diagnostic_;
    out.version = StateVersion{version_};
    out.time = time_;
    out.x_descriptor_ = field_descriptor(Field::x, FieldAccess::read);
    out.q_descriptor_ = field_descriptor(Field::q, FieldAccess::read);
    return out;
}
} // namespace nexora::reference
