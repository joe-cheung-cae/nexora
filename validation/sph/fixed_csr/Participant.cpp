#include <fixed_csr/Participant.hpp>

#include <array>
#include <cmath>
#include <data/CsrAdjacency.hpp>
#include <data/FieldIssuer.hpp>
#include <data/Permutation.hpp>
#include <exception>
#include <initializer_list>
#include <limits>
#include <fixed_csr/Density.hpp>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace nexora::sph_fixture {
namespace {
using detail::CsrAdjacency;
using detail::EntityOrder;
using detail::FieldAxis;
using detail::FieldIssuer;
using detail::Permutation;
using detail::ResourceBudget;
using detail::checked_array_bytes;
using detail::checked_csr_capacity;
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
template <class T> using View = Kokkos::View<T*, Kokkos::HostSpace>;

struct Failure {
    Status status;
};

template <class T> T require(Result<T> result) {
    if (!result.ok())
        throw Failure{result.status()};
    return std::move(result).value();
}

std::size_t payload_bytes(std::initializer_list<std::size_t> arrays) {
    std::size_t total = 0;
    for (const auto bytes : arrays) {
        if (bytes > std::numeric_limits<std::size_t>::max() - total)
            throw Failure{{StatusCode::invalid_argument, "SPH payload size exceeds capacity"}};
        total += bytes;
    }
    return total;
}

ResourceBudget::Ticket reserve_layout(const ResourceBudget& budget, std::size_t rows,
                                      std::size_t columns, std::size_t csr_bytes) {
    const auto row_bytes = require(checked_array_bytes(rows, sizeof(std::uint64_t) + sizeof(LocalIndex)));
    const auto column_bytes = require(checked_array_bytes(
        columns, sizeof(std::uint64_t) + 2 * sizeof(real) + 2 * sizeof(std::uint8_t)));
    return require(budget.reserve(payload_bytes({row_bytes, column_bytes, csr_bytes})));
}

template <class T>
View<T> allocate(const Kokkos::Serial& exec, const char* label, std::size_t size) {
    return View<T>(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, std::string(label)), size);
}

template <class T>
View<T> copy_input(const Kokkos::Serial& exec, const char* label, const std::vector<T>& input) {
    using Unmanaged = Kokkos::View<const T*, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    auto view = allocate<T>(exec, label, input.size());
    Kokkos::deep_copy(exec, view, Unmanaged(input.data(), input.size()));
    return view;
}

FieldDescriptor field(FieldId id, const char* name, const char* entity_set, const char* units,
                      FieldAccess access, FieldOwnership ownership, MeasureSemantics measure) {
    FieldDescriptor descriptor;
    descriptor.id = id;
    descriptor.name = name;
    descriptor.domain_id = DomainId{1};
    descriptor.association = Association::particle;
    descriptor.entity_set = entity_set;
    descriptor.units = units;
    descriptor.access = access;
    descriptor.ownership = ownership;
    descriptor.measure = measure;
    return descriptor;
}

FieldDescriptor density_field(FieldAccess access) {
    return field(FieldId{3}, "density", "sph_owned", "kg/m", access, FieldOwnership::owned,
                 MeasureSemantics::density);
}

ReadField<ConstView<real>> bind_read(ConstView<real> view, FieldDescriptor field, FieldStamp stamp) {
    stamp.storage_incarnation = 0;
    auto issuer = require(FieldIssuer::create(view, stamp));
    const std::array axes{FieldAxis{view.extent(0)}};
    return require(issuer.read(view, field, issuer.stamp(), {field.association, field.units, axes}));
}

// The supplied fixed graph defines neighborhood membership. Validate its
// entries and ownership without running an all-pairs neighbor search.
Result<View<LocalIndex>> check_neighborhood(const Kokkos::Serial& exec, const Config& config,
                                          const CsrAdjacency& adjacency) {
    const auto count = config.column_ids.size();
    const auto rows = config.row_ids.size();
    std::unordered_map<GlobalEntityId, LocalIndex> sources;
    sources.reserve(count);
    std::size_t owned = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(config.positions[i]) || !std::isfinite(config.masses[i]) ||
            config.masses[i] <= real{0} || config.ghost[i] > 1 || config.boundary[i] > 1)
            return Status{StatusCode::invalid_argument, "invalid SPH source position, mass or flag"};
        sources.emplace(config.column_ids[i], static_cast<LocalIndex>(i));
        owned += config.ghost[i] == 0;
    }
    if (rows != owned)
        return Status{StatusCode::invalid_argument, "SPH rows must cover the owned sources"};

    auto self = allocate<LocalIndex>(exec, "sph.self_columns", rows);
    std::vector<std::size_t> row_by_column(count, rows);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto source = sources.find(config.row_ids[row]);
        if (source == sources.end() || config.ghost[source->second] != 0)
            return Status{StatusCode::invalid_argument, "SPH row is missing its owned source"};
        self(row) = source->second;
        row_by_column[source->second] = row;
    }

    const auto offsets = adjacency.offsets();
    const auto indices = adjacency.indices();
    std::vector<std::size_t> last_row(count, rows);
    std::unordered_set<std::uint64_t> unmatched;
    for (std::size_t row = 0; row < rows; ++row) {
        for (Offset edge = offsets(row); edge < offsets(row + 1); ++edge) {
            const auto column = indices(edge);
            if (column == self(row) || last_row[column] == row ||
                !within_support(config.positions[self(row)], config.positions[column]))
                return Status{StatusCode::invalid_argument, "SPH neighbor is self, duplicate or outside support"};
            last_row[column] = row;
            const auto neighbor = row_by_column[column];
            if (neighbor == rows)
                continue; // A ghost does not have a destination row.
            if (row < neighbor) {
                unmatched.insert((static_cast<std::uint64_t>(row) << 32) | neighbor);
            } else if (!unmatched.erase((static_cast<std::uint64_t>(neighbor) << 32) | row)) {
                return Status{StatusCode::invalid_argument, "SPH owned edges must be reciprocal"};
            }
        }
    }
    if (!unmatched.empty())
        return Status{StatusCode::invalid_argument, "SPH owned edges must be reciprocal"};
    return self;
}

} // namespace

Snapshot::Geometry::Geometry(ResourceBudget::Ticket reservation, CsrAdjacency graph,
                             ConstView<LocalIndex> self_columns, ConstView<real> positions,
                             ConstView<real> masses, ConstView<std::uint8_t> is_ghost,
                             ConstView<std::uint8_t> is_boundary, const FieldStamp& stamp)
    : budget(std::move(reservation)), adjacency(std::move(graph)), self(std::move(self_columns)),
      x(bind_read(positions, field(FieldId{1}, "position", "sph_sources", "m", FieldAccess::read,
                                   FieldOwnership::replicated, MeasureSemantics::point_value), stamp)),
      mass(bind_read(masses, field(FieldId{2}, "mass", "sph_sources", "kg", FieldAccess::read,
                                   FieldOwnership::replicated, MeasureSemantics::point_value), stamp)),
      ghost(std::move(is_ghost)), boundary(std::move(is_boundary)) {}

struct Snapshot::Density {
    ResourceBudget::Ticket budget;
    ConstView<real> values;
};

Snapshot& Snapshot::operator=(Snapshot other) noexcept {
    // Destroy the old grants, payloads and record together in normal member order.
    using std::swap;
    swap(owned_density_sum, other.owned_density_sum);
    swap(stamp, other.stamp);
    swap(density_valid, other.density_valid);
    swap(snapshot_record_, other.snapshot_record_);
    swap(geometry_, other.geometry_);
    swap(density_, other.density_);
    swap(read_grant_, other.read_grant_);
    return *this;
}

std::size_t Snapshot::row_count() const { return geometry_ ? geometry_->adjacency.rows().size() : 0; }
std::size_t Snapshot::column_count() const { return geometry_ ? geometry_->adjacency.columns().size() : 0; }
std::size_t Snapshot::edge_count() const { return geometry_ ? geometry_->adjacency.indices().extent(0) : 0; }
Offset Snapshot::offset(std::size_t i) const { return geometry_->adjacency.offset_at(i); }
LocalIndex Snapshot::index(std::size_t i) const { return geometry_->adjacency.index_at(i); }
std::uint8_t Snapshot::ghost(std::size_t i) const { return geometry_->ghost(i); }
std::uint8_t Snapshot::boundary(std::size_t i) const { return geometry_->boundary(i); }
real Snapshot::density(std::size_t i) const { return density_->values(i); }

Snapshot Participant::layout_snapshot(const std::shared_ptr<const Geometry>& geometry) {
    Snapshot result;
    result.geometry_ = geometry;
    // Position owns a unique field incarnation even before density is available.
    result.stamp = geometry->x.stamp();
    return result;
}

Participant::Participant(Config config, ParticipantId id, ResourceLimits limits)
    : config_(std::move(config)), id_(id), budget_(limits) {}

Status Participant::initialize(double initial_time) {
    if (accepted_ || finalized_ || !Kokkos::is_initialized() || Kokkos::is_finalized())
        return {StatusCode::invalid_state, "SPH initialization requires a new participant and active Kokkos"};
    const auto count = config_.column_ids.size();
    if (!std::isfinite(initial_time) || config_.row_ids.empty() ||
        config_.positions.size() != count || config_.masses.size() != count ||
        config_.ghost.size() != count || config_.boundary.size() != count)
        return {StatusCode::invalid_argument, "invalid SPH initial time or field extents"};
    try {
        const auto rows_count = config_.row_ids.size();
        const auto edges = config_.indices.size();
        const auto csr_bytes = require(checked_csr_capacity(rows_count, count, edges));
        const auto reservation = reserve_layout(budget_, rows_count, count, csr_bytes);
        FieldStamp stamp;
        stamp.topology = TopologyVersion{1};
        stamp.geometry = GeometryVersion{1};
        stamp.layout = LayoutVersion{1};
        stamp.accepted = true;
        stamp.time.physical_time = stamp.valid_from = stamp.valid_until = initial_time;
        const auto& exec = exec_.emplace();
        const auto rows = require(EntityOrder::create(
            exec, {DomainId{1}, Association::particle, "sph_owned"}, stamp.layout, config_.row_ids));
        const auto columns = require(EntityOrder::create(
            exec, {DomainId{1}, Association::particle, "sph_sources"}, stamp.layout, config_.column_ids));
        auto adjacency = require(CsrAdjacency::create(
            exec, rows, columns, stamp.topology, config_.offsets, config_.indices));
        auto self = require(check_neighborhood(exec, config_, adjacency));
        auto positions = copy_input(exec, "sph.positions", config_.positions);
        auto masses = copy_input(exec, "sph.masses", config_.masses);
        auto ghost = copy_input(exec, "sph.ghost", config_.ghost);
        auto boundary = copy_input(exec, "sph.boundary", config_.boundary);
        exec.fence("SPH initial fields ready");
        auto geometry = std::make_shared<Geometry>(reservation, std::move(adjacency), self, positions, masses,
                                                   ghost, boundary, stamp);
        accepted_ = layout_snapshot(geometry);
        config_ = {};
        return {};
    } catch (const Failure& failure) {
        return failure.status;
    } catch (const std::bad_alloc&) {
        return {StatusCode::resource_exhausted, "cannot allocate SPH initial state"};
    } catch (const std::exception&) {
        return {StatusCode::execution_failure, "cannot initialize SPH state"};
    }
}

Status Participant::do_step(const TimeInterval& interval) {
    if (!accepted_)
        return {StatusCode::invalid_state, "SPH step requires an initialized participant"};
    if (interval.begin != accepted_->stamp.time.physical_time)
        return {StatusCode::invalid_argument, "SPH step does not start at the accepted time"};
    try {
        auto stamp = accepted_->stamp;
        stamp.accepted_version = require(detail::checked_next_version(stamp.accepted_version));
        stamp.storage_incarnation = 0;
        stamp.time = {interval.end, stamp.accepted_version.value(), 0, 0};
        stamp.valid_from = stamp.valid_until = interval.end;

        const auto count = accepted_->row_count();
        const auto reservation = require(budget_.reserve(require(checked_array_bytes(count, sizeof(real)))));
        const auto values = allocate<real>(*exec_, "sph.accepted_density", count);
        real sum{};
        {
            auto write_stamp = stamp;
            write_stamp.accepted = false;
            auto issuer = require(FieldIssuer::create(values, write_stamp));
            const std::array axes{FieldAxis{count}};
            const auto write = require(issuer.write(values, density_field(FieldAccess::overwrite),
                                                    issuer.stamp(), {Association::particle, "kg/m", axes},
                                                    FieldAccess::overwrite));
            const auto& geometry = *accepted_->geometry_;
            const DensityInput input{geometry.x.view(), geometry.mass.view(), geometry.adjacency.offsets(),
                                     geometry.adjacency.indices(), geometry.self};
            sum = require(evaluate_density(*exec_, input, write.view()));
        }
        auto candidate = layout_snapshot(accepted_->geometry_);
        auto grant = bind_read(values, density_field(FieldAccess::read), stamp);
        candidate.stamp = grant.stamp();
        candidate.density_ = std::make_shared<Snapshot::Density>(reservation, values);
        candidate.read_grant_ = std::make_shared<decltype(grant)>(std::move(grant));
        candidate.owned_density_sum = sum;
        candidate.density_valid = true;
        accepted_.emplace(std::move(candidate));
        return {};
    } catch (const Failure& failure) {
        return failure.status;
    } catch (const std::bad_alloc&) {
        return {StatusCode::resource_exhausted, "cannot allocate SPH density or its binding"};
    } catch (const std::exception&) {
        return {StatusCode::execution_failure, "cannot compute SPH density"};
    }
}

Status Participant::finalize() {
    if (finalized_)
        return {StatusCode::invalid_state, "SPH participant is finalized"};
    accepted_.reset();
    exec_.reset();
    config_ = {};
    finalized_ = true;
    return {};
}

Status Participant::reorder(std::span<const GlobalEntityId> row_ids,
                            std::span<const GlobalEntityId> column_ids, const FieldStamp& expected) {
    if (!accepted_ || accepted_->stamp != expected)
        return {StatusCode::invalid_state, "SPH reorder requires the current accepted layout"};
    // Charge the previous storage until the last local View used for reordering is released.
    const auto previous = *accepted_;
    const auto& old_geometry = *previous.geometry_;
    const auto& before = old_geometry.adjacency;
    if (row_ids.size() != before.rows().size() || column_ids.size() != before.columns().size())
        return {StatusCode::invalid_argument, "SPH reorder requires unchanged row and column counts"};
    try {
        auto stamp = previous.stamp;
        stamp.layout = require(detail::checked_next_version(stamp.layout));
        stamp.storage_incarnation = 0;
        const auto& exec = *exec_;
        const auto count = row_ids.size();
        const auto sources = column_ids.size();
        const auto edges = before.indices().extent(0);
        const bool has_density = previous.density_valid;
        const auto csr_bytes = require(checked_csr_capacity(count, sources, edges));
        const auto reservation = reserve_layout(budget_, count, sources, csr_bytes);
        ResourceBudget::Ticket density_ticket;
        if (has_density)
            density_ticket = require(budget_.reserve(count * sizeof(real)));
        // Matching retains two maps per axis. CSR construction copies the
        // rewritten offsets and indices into the promised immutable layout.
        const auto scratch_reservation = require(budget_.reserve(payload_bytes({
            require(checked_array_bytes(count, 2 * sizeof(LocalIndex))),
            require(checked_array_bytes(sources, 2 * sizeof(LocalIndex))), csr_bytes})));

        const auto rows = require(EntityOrder::create(exec, before.rows().entity_namespace(),
                                                       stamp.layout, row_ids));
        const auto columns = require(EntityOrder::create(exec, before.columns().entity_namespace(),
                                                          stamp.layout, column_ids));
        const auto row_map = require(Permutation::between(exec, before.rows(), rows));
        const auto column_map = require(Permutation::between(exec, before.columns(), columns));
        const auto row_forward = row_map.new_to_old();
        const auto column_forward = column_map.new_to_old();
        const auto column_inverse = column_map.old_to_new();

        const auto old_positions = old_geometry.x.view();
        const auto old_masses = old_geometry.mass.view();
        const auto old_ghost = old_geometry.ghost;
        const auto old_boundary = old_geometry.boundary;
        auto positions = allocate<real>(exec, "sph.reorder.positions", sources);
        auto masses = allocate<real>(exec, "sph.reorder.masses", sources);
        auto ghost = allocate<std::uint8_t>(exec, "sph.reorder.ghost", sources);
        auto boundary = allocate<std::uint8_t>(exec, "sph.reorder.boundary", sources);
        const auto old_self = old_geometry.self;
        const auto old_offsets = before.offsets();
        const auto old_indices = before.indices();
        const auto old_density = has_density ? previous.density_->values : ConstView<real>{};
        auto self = allocate<LocalIndex>(exec, "sph.reorder.self_columns", count);
        auto offsets = allocate<Offset>(exec, "sph.reorder.offsets", count + 1);
        auto indices = allocate<LocalIndex>(exec, "sph.reorder.indices", edges);
        auto density = has_density ? allocate<real>(exec, "sph.reorder.density", count) : Values{};

        Kokkos::parallel_for("sph reorder sources", Policy(exec, 0, columns.size()),
                             KOKKOS_LAMBDA(std::size_t i) {
            const auto from = column_forward(i);
            positions(i) = old_positions(from);
            masses(i) = old_masses(from);
            ghost(i) = old_ghost(from);
            boundary(i) = old_boundary(from);
        });
        // The Serial prefix preserves edge order as rows and columns move independently.
        Kokkos::parallel_for("sph reorder CSR offsets", Policy(exec, 0, 1),
                             KOKKOS_LAMBDA(std::size_t) {
            offsets(0) = 0;
            for (std::size_t i = 0; i < count; ++i) {
                const auto from = row_forward(i);
                offsets(i + 1) = offsets(i) + (old_offsets(from + 1) - old_offsets(from));
            }
        });
        Kokkos::parallel_for("sph reorder CSR and owned fields", Policy(exec, 0, count),
                             KOKKOS_LAMBDA(std::size_t i) {
            const auto from = row_forward(i);
            self(i) = column_inverse(old_self(from));
            if (has_density)
                density(i) = old_density(from);
            for (Offset edge = old_offsets(from); edge < old_offsets(from + 1); ++edge)
                indices(offsets(i) + (edge - old_offsets(from))) = column_inverse(old_indices(edge));
        });
        exec.fence("SPH reordered fields ready");
        auto adjacency = require(CsrAdjacency::create(
            exec, rows, columns, stamp.topology, {offsets.data(), offsets.extent(0)},
            {indices.data(), indices.extent(0)}));
        auto geometry = std::make_shared<Geometry>(reservation, std::move(adjacency), self, positions, masses,
                                                   ghost, boundary, stamp);
        auto snapshot = layout_snapshot(geometry);
        if (has_density) {
            auto grant = bind_read(density, density_field(FieldAccess::read), stamp);
            snapshot.stamp = grant.stamp();
            snapshot.density_ = std::make_shared<Snapshot::Density>(density_ticket, density);
            snapshot.owned_density_sum = previous.owned_density_sum;
            snapshot.density_valid = true;
            snapshot.read_grant_ = std::make_shared<decltype(grant)>(std::move(grant));
        }
        accepted_.emplace(std::move(snapshot));
        return {};
    } catch (const Failure& failure) {
        return failure.status;
    } catch (const std::bad_alloc&) {
        return {StatusCode::resource_exhausted, "cannot allocate SPH layout"};
    } catch (const std::exception&) {
        return {StatusCode::execution_failure, "cannot reorder SPH state"};
    }
}

Result<Snapshot> Participant::snapshot() const {
    if (!accepted_)
        return Status{StatusCode::invalid_state, "SPH has no accepted state"};
    auto record = budget_.reserve(0, 1);
    if (!record.ok())
        return record.status();
    auto result = *accepted_;
    result.snapshot_record_ = std::move(record).value();
    return result;
}

} // namespace nexora::sph_fixture
