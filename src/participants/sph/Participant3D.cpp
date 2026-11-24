#include "Participant3D.hpp"

#include <array>
#include <cmath>
#include <data/EntityCapacity.hpp>
#include <data/FieldIssuer.hpp>
#include <limits>
#include <methods/sph/Density3D.hpp>
#include <new>
#include <string>

namespace nexora::sph {
namespace {

using Ticket = detail::ResourceBudget::Ticket;
using SphBoundary::FixedWall3D;
using ScalarRead = ReadField<Values::const_type>;
using AccelerationRead = ReadField<Acceleration3D::const_type>;
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;

struct Failure {
    Status status;
};

template <class T> T require(Result<T> result) {
    if (!result.ok())
        throw Failure{result.status()};
    return std::move(result).value();
}

template <class F> auto guarded(F&& operation) -> decltype(operation()) {
    try {
        return operation();
    } catch (const Failure& error) {
        return error.status;
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "cannot allocate 3D SPH participant state"};
    } catch (const std::exception&) {
        return Status{StatusCode::execution_failure, "3D SPH participant operation failed"};
    }
}

FieldDescriptor particle_field(FieldId id, const char* name, const char* units,
                               std::size_t components, FieldAccess access) {
    FieldDescriptor field;
    field.id = id;
    field.name = name;
    field.domain_id = DomainId{1};
    field.association = Association::particle;
    field.entity_set = "sph_owned";
    field.units = units;
    field.components = components;
    field.ownership = FieldOwnership::owned;
    field.access = access;
    return field;
}

FieldDescriptor density_field(FieldAccess access) {
    auto field = particle_field(FieldId{3}, "density", "kg/m^3", 1, access);
    field.measure = MeasureSemantics::density;
    return field;
}

FieldDescriptor pressure_field(FieldAccess access) {
    auto field = particle_field(FieldId{4}, "pressure", "Pa", 1, access);
    field.pressure_convention = "gauge";
    return field;
}

FieldDescriptor acceleration_field(FieldAccess access) {
    auto field = particle_field(FieldId{5}, "pressure_acceleration", "m/s^2", 3, access);
    field.reference_frame = "grid_cartesian";
    field.tensor_interpretation = "vector";
    return field;
}

FieldDescriptor velocity_field(FieldAccess access) {
    auto field = particle_field(FieldId{6}, "velocity", "m/s", 3, access);
    field.reference_frame = "grid_cartesian";
    field.tensor_interpretation = "vector";
    return field;
}

template <class View> auto owned_axes(const View& values) {
    if constexpr (View::rank() == 1)
        return std::array{detail::FieldAxis{values.extent(0)}};
    else
        return std::array{detail::FieldAxis{values.extent(0)}, detail::FieldAxis{3, true}};
}

template <class View>
auto read_field(const View& values, FieldDescriptor descriptor, FieldStamp stamp) {
    stamp.storage_incarnation = 0;
    auto issuer = require(detail::FieldIssuer::create(values, stamp));
    const auto axes = owned_axes(values);
    const detail::FieldBindingSpec spec{Association::particle, descriptor.units, axes};
    return require(issuer.read(values, descriptor, issuer.stamp(), spec));
}

template <class View>
auto write_field(const View& values, FieldDescriptor descriptor, FieldStamp stamp) {
    stamp.accepted = false;
    stamp.storage_incarnation = 0;
    auto issuer = require(detail::FieldIssuer::create(values, stamp));
    const auto axes = owned_axes(values);
    const detail::FieldBindingSpec spec{Association::particle, descriptor.units, axes};
    return require(issuer.write(values, descriptor, issuer.stamp(), spec, FieldAccess::overwrite));
}

template <class View>
View allocate(const Kokkos::Serial& exec, const char* label, std::size_t rows) {
    return View(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, std::string(label)), rows);
}

template <class View>
auto copy_field(const Kokkos::Serial& exec, const View& source, const char* label) {
    auto copy = allocate<typename View::non_const_type>(exec, label, source.extent(0));
    Kokkos::deep_copy(exec, copy, source);
    return View(copy);
}

template <class... Views> bool source_extents(std::size_t n, const Views&... views) {
    return ((views.extent(0) == n && (!n || views.data())) && ...);
}

struct CombinedWallSources {
    Ticket reservation;
    ParticleFields3D fields;
};

CombinedWallSources combine_fixed_walls(const Kokkos::Serial& exec, const ParticleFields3D& fluid,
                                        const FixedWall3D& walls,
                                        const detail::ResourceBudget& budget) {
    const auto nf = fluid.position.extent(0);
    const auto nw = walls.particles().extent(0);
    if (!source_extents(nf, fluid.position, fluid.mass, fluid.ids, fluid.ghost))
        throw Failure{{StatusCode::invalid_argument, "invalid 3D SPH fluid fields"}};
    std::size_t ghosts{};
    Kokkos::parallel_reduce(
        "sph.wall.validate_ownership", Policy(exec, 0, nf),
        [&](std::size_t i, std::size_t& count) { count += fluid.ghost(i) != 0; }, ghosts);
    if (ghosts)
        throw Failure{
            {StatusCode::unsupported, "3D SPH fixed walls require all-owned fluid input"}};
    if (!nw)
        return {{}, fluid};
    const auto n = nf + nw;
    auto ticket = require(budget.reserve(require(detail::checked_array_bytes(
        n, 4 * sizeof(real) + sizeof(std::uint64_t) + sizeof(std::uint8_t)))));
    auto position = allocate<Velocity3D>(exec, "sph.wall.combined_position", n);
    auto mass = allocate<Values>(exec, "sph.wall.combined_mass", n);
    auto ids = allocate<decltype(fluid.ids)::non_const_type>(exec, "sph.wall.combined_ids", n);
    auto flags =
        allocate<decltype(fluid.ghost)::non_const_type>(exec, "sph.wall.combined_flags", n);
    Kokkos::parallel_for("sph.wall.combine_sources", Policy(exec, 0, n), [&](std::size_t i) {
        const bool wall = i >= nf;
        if (wall) {
            const auto& particle = walls.particles()(i - nf);
            const auto sample = walls.sample(particle.id, 0).value();
            for (int axis = 0; axis < 3; ++axis)
                position(i, axis) = sample.position[axis];
            mass(i) = particle.mass;
            ids(i) = particle.id;
        } else {
            for (int axis = 0; axis < 3; ++axis)
                position(i, axis) = fluid.position(i, axis);
            mass(i) = fluid.mass(i);
            ids(i) = fluid.ids(i);
        }
        flags(i) = wall;
    });
    exec.fence("3D SPH combined fluid and wall inputs ready");
    return {std::move(ticket), {position, mass, ids, flags}};
}

template <class Write>
void sample_fixed_walls(const Kokkos::Serial& exec, const FixedWall3D& walls,
                        const ParticleFields3D& fields, const char* label, const Write& write) {
    std::size_t invalid{};
    Kokkos::parallel_reduce(
        label, Policy(exec, 0, fields.ids.extent(0)),
        [&](std::size_t slot, std::size_t& errors) {
            if (!fields.ghost(slot))
                return;
            const auto sample = walls.sample(fields.ids(slot), 0);
            if (sample.ok())
                write(slot, sample.value());
            else
                ++errors;
        },
        invalid);
    if (invalid)
        throw Failure{{StatusCode::invalid_state, "fixed wall ID missing from 3D SPH geometry"}};
}

std::size_t field_bytes(std::size_t rows) {
    // Density, pressure, pressure acceleration, and velocity follow the same owned rows.
    return require(detail::checked_array_bytes(rows, (1 + 1 + 3 + 3) * sizeof(real)));
}

} // namespace

struct Snapshot3D::Fields {
    Ticket reservation;
    ScalarRead density, pressure;
    AccelerationRead acceleration, velocity;
    double maximum_step;
};

real Snapshot3D::density(std::size_t row) const {
    return fields_->density.view()(row);
}

Result<real> Snapshot3D::pressure(std::size_t row) const {
    if (row >= row_count())
        return Status{StatusCode::invalid_argument, "3D SPH pressure row is out of range"};
    return fields_->pressure.view()(row);
}

Result<std::array<real, 3>> Snapshot3D::pressure_acceleration(std::size_t row) const {
    if (row >= row_count())
        return Status{StatusCode::invalid_argument,
                      "3D SPH pressure acceleration row is out of range"};
    const auto& values = fields_->acceleration.view();
    return std::array{values(row, 0), values(row, 1), values(row, 2)};
}

Result<std::array<real, 3>> Snapshot3D::position(std::size_t row) const {
    if (row >= row_count())
        return Status{StatusCode::invalid_argument, "3D SPH position row is out of range"};
    const auto& grid = geometry_->grid;
    const auto slot = grid.owned_slots()(row);
    const auto& values = grid.fields().position;
    return std::array{values(slot, 0), values(slot, 1), values(slot, 2)};
}

Result<std::array<real, 3>> Snapshot3D::velocity(std::size_t row) const {
    if (row >= row_count())
        return Status{StatusCode::invalid_argument, "3D SPH velocity row is out of range"};
    const auto values = fields_->velocity.view();
    return std::array{values(row, 0), values(row, 1), values(row, 2)};
}

Participant3D::Participant3D(ParticleFields3D initial, std::size_t max_cells,
                             WcsphVerletParameters parameters, Velocity3D::const_type velocity,
                             ParticipantId id, ResourceLimits limits)
    : initial_(std::move(initial)), max_cells_(max_cells), id_(id), budget_(limits),
      parameters_(parameters), initial_velocity_(std::move(velocity)) {}

Participant3D::Participant3D(ParticleFields3D fluid, std::size_t max_cells,
                             WcsphVerletParameters parameters, Velocity3D::const_type velocity,
                             FixedWallParticles3D walls, ParticipantId id, ResourceLimits limits)
    : Participant3D(std::move(fluid), max_cells, parameters, std::move(velocity), id, limits) {
    initial_walls_.emplace(std::move(walls));
}

Participant3D::Participant3D(ParticleFields3D fluid, std::size_t max_cells,
                             WcsphVerletParameters parameters, Velocity3D::const_type velocity,
                             Values::const_type density, ParticipantId id, ResourceLimits limits)
    : Participant3D(std::move(fluid), max_cells, parameters, std::move(velocity), id, limits) {
    initial_density_.emplace(std::move(density));
}

Participant3D::Participant3D(ParticleFields3D fluid, std::size_t max_cells,
                             WcsphVerletParameters parameters, Velocity3D::const_type velocity,
                             Values::const_type density, FixedWallParticles3D walls,
                             ParticipantId id, ResourceLimits limits)
    : Participant3D(std::move(fluid), max_cells, parameters, std::move(velocity),
                    std::move(density), id, limits) {
    initial_walls_.emplace(std::move(walls));
}

TimeConstraint Participant3D::time_constraint(double) const {
    TimeConstraint result;
    if (accepted_)
        result.maximum_step = accepted_->fields_->maximum_step;
    return result;
}

Snapshot3D Participant3D::compute(std::shared_ptr<const Geometry3D> geometry, FieldStamp stamp,
                                  const Velocity3D::const_type& input_velocity,
                                  const FixedWall3D* walls, double dt,
                                  const Values::const_type* owned_density,
                                  IntegrationWorkspace* finished_continuity) const {
    stamp.topology = geometry->topology;
    stamp.geometry = geometry->geometry;
    stamp.layout = geometry->layout;
    const auto rows = geometry->grid.owned_slots().extent(0);
    const auto reserve = [&](std::size_t bytes) {
        auto reservation = budget_.reserve(bytes);
        if (!reservation.ok() && finished_continuity && finished_continuity->stage) {
            // The two rates, row mapping and stage velocity have had their last read.
            finished_continuity->continuity.reset();
            finished_continuity->stage.reset();
            reservation = budget_.reserve(bytes);
        }
        return require(std::move(reservation));
    };
    auto ticket = reserve(field_bytes(rows));
    auto density = allocate<Values>(*exec_, "sph.participant3d.density", rows);
    {
        auto write = write_field(density, density_field(FieldAccess::overwrite), stamp);
        if (owned_density) {
            Kokkos::deep_copy(*exec_, write.view(), *owned_density);
        } else if (initial_density_) {
            const auto owned = geometry->grid.owned_slots();
            const auto input = geometry->grid.sorted_to_input();
            const auto out = write.view();
            const auto seed = *initial_density_;
            Kokkos::parallel_for("sph.continuity.initial_density", Policy(*exec_, 0, rows),
                                 [&](std::size_t row) { out(row) = seed(input(owned(row))); });
        } else {
            const auto evaluated = evaluate_density_3d(*exec_, geometry->grid.neighbors(), write.view());
            if (!evaluated.ok())
                throw Failure{evaluated};
        }
    }
    auto density_read =
        read_field(Values::const_type(density), density_field(FieldAccess::read), stamp);
    auto pressure = allocate<Values>(*exec_, "sph.participant3d.pressure", rows);
    auto acceleration =
        allocate<Acceleration3D>(*exec_, "sph.participant3d.pressure_acceleration", rows);
    {
        Ticket workspace;
        WallAcceleration3D wall_acceleration;
        const auto& sources = geometry->grid.fields();
        const auto count = sources.ids.extent(0);
        const bool sample_acceleration = walls && rows && count != rows;
        if (rows != 0) {
            auto bytes = pressure_workspace_bytes_per_source(count != rows, walls != nullptr);
            if (sample_acceleration)
                bytes += 3 * sizeof(double);
            workspace = reserve(require(detail::checked_array_bytes(count, bytes)));
        }
        if (sample_acceleration) {
            wall_acceleration = allocate<WallAcceleration3D>(*exec_, "sph.wall.acceleration", count);
            sample_fixed_walls(*exec_, *walls, sources, "sph.wall.sample_acceleration",
                               [&](std::size_t slot, const auto& sample) {
                                   for (int axis = 0; axis < 3; ++axis)
                                       wall_acceleration(slot, axis) = sample.acceleration[axis];
                               });
        }
        auto pressure_write = write_field(pressure, pressure_field(FieldAccess::overwrite), stamp);
        auto acceleration_write =
            write_field(acceleration, acceleration_field(FieldAccess::overwrite), stamp);
        const TaitEos eos(parameters_.eos);
        auto evaluated = walls
            ? evaluate_pressure_3d(*exec_, geometry->grid.neighbors(), eos, density_read.view(),
                                   pressure_write.view(), acceleration_write.view(),
                                   wall_acceleration, parameters_.gravity)
            : evaluate_pressure_3d(*exec_, geometry->grid.neighbors(), eos, density_read.view(),
                                   pressure_write.view(), acceleration_write.view());
        if (!evaluated.ok())
            throw Failure{std::move(evaluated)};
    }
    auto pressure_read =
        read_field(Values::const_type(pressure), pressure_field(FieldAccess::read), stamp);
    auto acceleration_read = read_field(Acceleration3D::const_type(acceleration),
                                        acceleration_field(FieldAccess::read), stamp);
    auto velocity = allocate<Velocity3D>(*exec_, "sph.participant3d.velocity", rows);
    {
        auto write = write_field(velocity, velocity_field(FieldAccess::overwrite), stamp);
        if (dt > 0) {
            const auto status =
                finish_kick_3d(*exec_, geometry->grid, input_velocity, acceleration_read.view(), dt,
                               write.view(), parameters_.gravity);
            if (!status.ok())
                throw Failure{status};
        } else {
            const auto to_input = geometry->grid.sorted_to_input();
            const auto owned = geometry->grid.owned_slots();
            const auto out = write.view();
            std::size_t invalid{};
            Kokkos::parallel_reduce(
                "sph.verlet.initial_velocity", Policy(*exec_, 0, rows),
                [&](std::size_t row, std::size_t& count) {
                    for (int axis = 0; axis < 3; ++axis) {
                        const auto value = input_velocity(to_input(owned(row)), axis);
                        out(row, axis) = value;
                        count += !Kokkos::isfinite(value);
                    }
                },
                invalid);
            if (invalid != 0)
                throw Failure{
                    {StatusCode::invalid_argument, "3D SPH initial velocity must be finite"}};
        }
    }
    auto velocity_read =
        read_field(Velocity3D::const_type(velocity), velocity_field(FieldAccess::read), stamp);
    const auto maximum_step =
        require(cfl_step_3d(*exec_, velocity_read.view(), parameters_.eos.sound_speed,
                            parameters_.cfl, geometry->grid.smoothing_length()));
    Snapshot3D result;
    result.stamp = density_read.stamp();
    result.geometry_ = std::move(geometry);
    result.fields_ = std::make_shared<Snapshot3D::Fields>(
        std::move(ticket), std::move(density_read), std::move(pressure_read),
        std::move(acceleration_read), std::move(velocity_read), maximum_step);
    return result;
}

Snapshot3D Participant3D::integrate(FieldStamp stamp, double dt) {
    // Hold scratch locally until the trial succeeds, so failures release its budget.
    auto work = std::exchange(integration_workspace_, {});
    const auto& before = accepted_->geometry_->grid;
    const auto sources = before.fields().ids.extent(0);
    const auto source_vector = [&](const char* label) {
        return IntegrationWorkspace::Buffer<Velocity3D>{
            require(budget_.reserve(require(detail::checked_array_bytes(sources, 3 * sizeof(real))))),
            allocate<Velocity3D>(*exec_, label, sources)};
    };
    if (!work.half_velocity)
        work.half_velocity.emplace(source_vector("sph.verlet.half_velocity"));
    if (!work.stage)
        work.stage.emplace(source_vector("sph.verlet.stage"));
    const auto& half_velocity = work.half_velocity->values;
    const auto* walls = fixed_walls_ ? &fixed_walls_->boundary : nullptr;
    auto geometry = [&] {
        const auto& position = work.stage->values;
        const auto& fields = *accepted_->fields_;
        const auto kicked =
            kick_drift_3d(*exec_, before, fields.velocity.view(), fields.acceleration.view(), dt,
                          position, half_velocity, parameters_.gravity);
        if (!kicked.ok())
            throw Failure{kicked};
        const auto& old_fields = before.fields();
        if (walls)
            sample_fixed_walls(*exec_, *walls, old_fields, "sph.wall.sample_position",
                               [&](std::size_t slot, const auto& sample) {
                                   for (int axis = 0; axis < 3; ++axis)
                                       position(slot, axis) = sample.position[axis];
                               });
        const ParticleFields3D updated{position, old_fields.mass, old_fields.ids, old_fields.ghost};
        auto next = prepare_geometry_3d(*exec_, updated, max_cells_, budget_,
                                        parameters_.smoothing_length, accepted_->geometry_.get());
        // Only the positions change between these two grids.
        if (!next.ok() && next.status().code() == StatusCode::invalid_argument)
            throw Failure{{StatusCode::numerical_failure, "invalid drifted WCSPH position"}};
        return require(std::move(next));
    }();
    const auto& refreshed = geometry->grid;
    const auto old_owned = before.owned_slots();
    const auto new_owned = refreshed.owned_slots();
    const auto rows = old_owned.extent(0);
    // Each successful grid path has finished gathering independent packed positions.
    // Continuity can now use the same source buffer for velocity.
    if (!work.corrected) {
        work.corrected.emplace(IntegrationWorkspace::Buffer<Values>{
            require(budget_.reserve(require(detail::checked_array_bytes(rows, sizeof(real))))),
            allocate<Values>(*exec_, "sph.continuity.density", rows)});
    }
    const auto& corrected = work.corrected->values;
    {
        if (!work.continuity) {
            work.continuity.emplace(IntegrationWorkspace::Continuity{
                require(budget_.reserve(
                    require(detail::checked_array_bytes(sources, sizeof(LocalIndex))))),
                require(budget_.reserve(
                    require(detail::checked_array_bytes(rows, 2 * sizeof(double))))),
                allocate<CellGrid::Indices::non_const_type>(*exec_, "sph.continuity.old_row", sources),
                allocate<DensityRate3D>(*exec_, "sph.continuity.accepted_rate", rows),
                allocate<DensityRate3D>(*exec_, "sph.continuity.predicted_rate", rows)});
        }
        const auto& velocity = work.stage->values;
        const auto& old_row = work.continuity->old_row;
        const auto& rate_n = work.continuity->rate_n;
        const auto& rate_star = work.continuity->rate_star;
        const auto v = accepted_->fields_->velocity.view();
        const auto rho = accepted_->fields_->density.view();
        const auto acceleration = accepted_->fields_->acceleration.view();
        const Policy policy(*exec_, 0, rows);
        Kokkos::parallel_for("sph.continuity.accepted_velocity", policy, [&](std::size_t row) {
            const auto slot = old_owned(row);
            old_row(slot) = static_cast<LocalIndex>(row);
            for (int axis = 0; axis < 3; ++axis)
                velocity(slot, axis) = v(row, axis);
        });
        const auto write_wall_velocity = [&](std::size_t slot, const auto& sample) {
            for (int axis = 0; axis < 3; ++axis)
                velocity(slot, axis) = sample.velocity[axis];
        };
        if (walls)
            sample_fixed_walls(*exec_, *walls, before.fields(), "sph.wall.sample_old_velocity",
                               write_wall_velocity);
        auto status = continuity_rate_3d(*exec_, before.neighbors(), velocity, rate_n);
        if (!status.ok())
            throw Failure{status};
        const auto to_new = refreshed.input_to_sorted();
        std::size_t invalid{};
        Kokkos::parallel_reduce(
            "sph.continuity.predict_velocity", policy,
            [&](std::size_t row, std::size_t& errors) {
                const auto slot = to_new(old_owned(row));
                for (int axis = 0; axis < 3; ++axis) {
                    const double value =
                        Kokkos::fma(dt, acceleration(row, axis) + parameters_.gravity[axis],
                                    double(v(row, axis)));
                    if (Kokkos::abs(value) <= std::numeric_limits<real>::max())
                        velocity(slot, axis) = static_cast<real>(value);
                    else
                        ++errors;
                }
            },
            invalid);
        if (invalid)
            throw Failure{{StatusCode::numerical_failure, "invalid predicted WCSPH velocity"}};
        if (walls)
            sample_fixed_walls(*exec_, *walls, refreshed.fields(), "sph.wall.sample_new_velocity",
                               write_wall_velocity);
        status = continuity_rate_3d(*exec_, refreshed.neighbors(), velocity, rate_star);
        if (!status.ok())
            throw Failure{status};
        const auto to_old = refreshed.sorted_to_input();
        // Refreshed input slots are accepted sorted slots. This permutation transports
        // both accepted density and accepted rate with the same particle through a rebuild.
        Kokkos::parallel_for("sph.continuity.correct_density", policy, [&](std::size_t row) {
            const auto previous = old_row(to_old(new_owned(row)));
            corrected(row) = narrow_step_value(Kokkos::fma(
                .5 * dt, rate_n(previous) + rate_star(row), double(rho(previous))));
        });
    }
    const Values::const_type completed = corrected;
    auto result = compute(std::move(geometry), stamp, half_velocity, walls, dt, &completed, &work);
    integration_workspace_ = std::move(work);
    return result;
}

Status Participant3D::initialize(double time) {
    if (stage_ != Stage::created || !Kokkos::is_initialized() || Kokkos::is_finalized())
        return {StatusCode::invalid_state,
                "3D SPH initialization requires a new participant and active Kokkos"};
    if (!std::isfinite(time))
        return {StatusCode::invalid_argument, "invalid 3D SPH initial time"};
    return guarded([&]() -> Status {
        const auto valid = parameters_.validate();
        if (!valid.ok())
            return valid;
        const auto& exec = exec_.emplace();
        const auto count = initial_.position.extent(0);
        if (initial_density_ && !source_extents(count, *initial_density_))
            return {StatusCode::invalid_argument, "3D SPH initial density extent mismatch"};
        if (!source_extents(count, initial_velocity_))
            return {StatusCode::invalid_argument, "3D SPH initial velocity extent mismatch"};
        std::optional<FixedWallState> walls;
        if (initial_walls_) {
            const auto nw = initial_walls_->ids.extent(0);
            if (!source_extents(nw, initial_walls_->position, initial_walls_->mass,
                                initial_walls_->ids))
                return {StatusCode::invalid_argument, "invalid 3D SPH wall fields"};
            if (nw > std::numeric_limits<std::size_t>::max() - count ||
                !detail::checked_entity_count(count + nw).ok())
                return {StatusCode::resource_exhausted, "3D SPH fluid and wall count exceeds capacity"};
            Ticket reservation;
            if (nw)
                reservation = require(budget_.reserve(require(detail::checked_array_bytes(
                    nw, sizeof(SphBoundary::WallParticle3D)))));
            walls.emplace(FixedWallState{
                std::move(reservation),
                FixedWall3D(require(SphBoundary::capture_wall_particles(exec, *initial_walls_)))});
        }
        auto geometry = [&] {
            if (walls) {
                auto combined = combine_fixed_walls(exec, initial_, walls->boundary, budget_);
                return require(prepare_geometry_3d(exec, combined.fields, max_cells_, budget_,
                                                   parameters_.smoothing_length));
            }
            return require(prepare_geometry_3d(exec, initial_, max_cells_, budget_,
                                               parameters_.smoothing_length));
        }();
        if (!walls && geometry->grid.owned_slots().extent(0) != geometry->grid.fields().ids.extent(0))
            return {StatusCode::unsupported, "3D SPH Verlet integration does not support ghosts"};
        FieldStamp stamp;
        stamp.accepted = true;
        stamp.time.physical_time = stamp.valid_from = stamp.valid_until = time;
        accepted_.emplace(compute(std::move(geometry), stamp, initial_velocity_,
                                  walls ? &walls->boundary : nullptr));
        fixed_walls_ = std::move(walls);
        initial_ = {};
        initial_velocity_ = {};
        // Release caller storage. Engagement records the explicit-density integration mode,
        // including the valid empty-fluid case.
        if (initial_density_)
            *initial_density_ = {};
        initial_walls_.reset();
        stage_ = Stage::initialized;
        return {};
    });
}

Status Participant3D::prepare() {
    if (stage_ != Stage::initialized)
        return {StatusCode::invalid_state, "3D SPH is not initialized"};
    stage_ = Stage::ready;
    return {};
}

Status Participant3D::begin_window(const TimeInterval& interval) {
    if (stage_ != Stage::ready)
        return {StatusCode::invalid_state, "3D SPH window requires a prepared participant"};
    if (!initial_density_)
        return {StatusCode::unsupported, "WCSPH integration requires explicit initial fluid density"};
    if (auto status = validate_time_step(time_constraint(interval.begin), interval.duration());
        !status.ok())
        return status;
    if (interval.duration() > accepted_->fields_->maximum_step)
        return {StatusCode::invalid_argument, "requested 3D SPH step exceeds accepted CFL bound"};
    if (interval.begin != accepted_->stamp.time.physical_time || !std::isfinite(interval.end) ||
        interval.end <= interval.begin || interval.end != interval.begin + interval.duration())
        return {StatusCode::invalid_argument, "invalid 3D SPH window interval"};
    if (attempt_ == std::numeric_limits<std::uint64_t>::max() ||
        !detail::checked_next_version(accepted_->stamp.accepted_version).ok())
        return {StatusCode::resource_exhausted, "3D SPH attempt or accepted version exhausted"};
    ++attempt_;
    interval_ = interval;
    stage_ = Stage::open;
    return {};
}

Status Participant3D::advance_trial(const TimeInterval& interval) {
    if (stage_ != Stage::open || !matches(interval))
        return Status{StatusCode::invalid_state, "no matching open 3D SPH window"};
    stage_ = Stage::failed;
    return guarded([&]() -> Status {
        auto stamp = accepted_->stamp;
        stamp.accepted = false;
        stamp.time = {interval.end, stamp.accepted_version.value() + 1, 0, 0};
        stamp.valid_from = stamp.valid_until = interval.end;
        stamp.trial = {stamp.time.window, attempt_, 0, 0};
        trial_.emplace(integrate(stamp, interval.duration()));
        stage_ = Stage::advanced;
        return {};
    });
}

Status Participant3D::prepare_commit(const TimeInterval& interval) {
    if (stage_ != Stage::advanced || !matches(interval))
        return {StatusCode::invalid_state, "no matching advanced 3D SPH window"};
    stage_ = Stage::failed;
    return guarded([&]() -> Status {
        Snapshot3D candidate;
        candidate.geometry_ = trial_->geometry_;
        auto stamp = trial_->stamp;
        stamp.accepted = true;
        stamp.trial = {};
        stamp.accepted_version = StateVersion{stamp.time.window};
        Ticket ticket;
        auto density = trial_->fields_->density.view();
        auto pressure = trial_->fields_->pressure.view();
        auto acceleration = trial_->fields_->acceleration.view();
        auto velocity = trial_->fields_->velocity.view();
        const auto maximum_step = trial_->fields_->maximum_step;
        if (trial_->fields_.use_count() == 1) {
            // Release all private trial grants before rebinding their allocations.
            ticket = trial_->fields_->reservation;
            trial_->fields_.reset();
        } else {
            // A retained trial keeps every allocation, binding and stamp.
            const auto bytes = field_bytes(density.extent(0));
            auto reservation = budget_.reserve(bytes);
            if (!reservation.ok() && integration_workspace_.half_velocity) {
                integration_workspace_ = {};
                reservation = budget_.reserve(bytes);
            }
            ticket = require(std::move(reservation));
            density = copy_field(*exec_, density, "sph.participant3d.accepted_density");
            pressure = copy_field(*exec_, pressure, "sph.participant3d.accepted_pressure");
            acceleration =
                copy_field(*exec_, acceleration, "sph.participant3d.accepted_acceleration");
            velocity = copy_field(*exec_, velocity, "sph.participant3d.accepted_velocity");
            exec_->fence("3D SPH retained trial fields copied");
        }
        auto read = read_field(density, density_field(FieldAccess::read), stamp);
        candidate.stamp = read.stamp();
        candidate.fields_ = std::make_shared<Snapshot3D::Fields>(
            std::move(ticket), std::move(read),
            read_field(pressure, pressure_field(FieldAccess::read), stamp),
            read_field(acceleration, acceleration_field(FieldAccess::read), stamp),
            read_field(velocity, velocity_field(FieldAccess::read), stamp), maximum_step);
        pending_.emplace(std::move(candidate));
        stage_ = Stage::precommitted;
        return {};
    });
}

Status Participant3D::commit_window(const TimeInterval& interval) {
    if (stage_ != Stage::precommitted || !matches(interval))
        return {StatusCode::invalid_state, "no matching prepared 3D SPH commit"};
    accepted_ = std::move(pending_);
    close_window();
    return {};
}

Status Participant3D::restore_window() {
    if (stage_ != Stage::open && stage_ != Stage::advanced && stage_ != Stage::precommitted &&
        stage_ != Stage::failed)
        return {StatusCode::invalid_state, "no 3D SPH window to restore"};
    close_window();
    return {};
}

Status Participant3D::finalize() {
    if (stage_ == Stage::finalized)
        return {StatusCode::invalid_state, "3D SPH has already finalized"};
    close_window();
    accepted_.reset();
    initial_ = {};
    initial_velocity_ = {};
    initial_density_.reset();
    initial_walls_.reset();
    fixed_walls_.reset();
    integration_workspace_ = {};
    exec_.reset();
    stage_ = Stage::finalized;
    return {};
}

Result<Snapshot3D> Participant3D::issue(const Snapshot3D& source) const {
    auto record = budget_.reserve(0, 1);
    if (!record.ok())
        return record.status();
    auto result = source;
    result.record_ = std::move(record).value();
    return result;
}
Result<Snapshot3D> Participant3D::snapshot() const {
    if (!accepted_)
        return Status{StatusCode::invalid_state, "3D SPH has no accepted state"};
    return issue(*accepted_);
}
Result<FieldStamp> Participant3D::trial_stamp() const {
    if (stage_ != Stage::advanced)
        return Status{StatusCode::invalid_state, "3D SPH has no readable trial"};
    return trial_->stamp;
}
Result<Snapshot3D> Participant3D::trial_snapshot(const FieldStamp& expected) const {
    if (stage_ != Stage::advanced || trial_->stamp != expected)
        return Status{StatusCode::invalid_state, "3D SPH trial stamp is stale or unavailable"};
    return issue(*trial_);
}
bool Participant3D::matches(const TimeInterval& interval) const {
    return interval.begin == interval_.begin && interval.end == interval_.end &&
           interval.duration() == interval_.duration();
}
void Participant3D::close_window() noexcept {
    pending_.reset();
    trial_.reset();
    interval_ = {};
    stage_ = Stage::ready;
}

} // namespace nexora::sph
