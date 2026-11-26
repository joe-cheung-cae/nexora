#include "WcsphStep3D.hpp"

#include <algorithm>
#include <data/EntityCapacity.hpp>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

namespace nexora::sph {
namespace {
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
using Budget = detail::ResourceBudget;
using Ticket = Budget::Ticket;
using namespace SphBoundary;
struct Failure { Status status; };

void require(Status status) {
    if (!status.ok())
        throw Failure{std::move(status)};
}
template <class T> T require(Result<T> result) {
    if (!result.ok())
        throw Failure{result.status()};
    return std::move(result).value();
}
template <class F> Status guarded(F&& operation) {
    try {
        operation();
        return {};
    } catch (const Failure& f) {
        return f.status;
    } catch (const std::bad_alloc&) {
        return {StatusCode::resource_exhausted, "cannot allocate WCSPH step storage"};
    } catch (const std::exception&) {
        return {StatusCode::execution_failure, "WCSPH step operation failed"};
    }
}

template <class V> V allocate(const Kokkos::Serial& exec, const char* label, std::size_t n) {
    return V(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, std::string(label)), n);
}

template <class... V> bool valid_extent(std::size_t n, const V&... fields) {
    return ((fields.extent(0) == n && (!n || fields.data())) && ...);
}

void stamp_geometry(WcsphState3D& state) {
    state.stamp.topology = state.geometry.topology;
    state.stamp.geometry = state.geometry.geometry;
    state.stamp.layout = state.geometry.layout;
}

// An initial copy is needed only for wall assembly or when grid packing changes
// fluid input order. All four input arrays in this copy share one lifetime.
struct InitialInput {
    Ticket ticket;
    ParticleFields3D fields;
};

template <class CaptureWalls>
InitialInput capture_input(const Kokkos::Serial& exec, const Budget& budget,
                           const ParticleFields3D& fluid, std::size_t nw, const CaptureWalls& capture_walls) {
    const auto nf = fluid.ids.extent(0);
    const auto n = nf + nw; // Bounded by initial source admission.
    auto ticket = require(budget.reserve(
        n * (4 * sizeof(real) + sizeof(std::uint64_t) + sizeof(std::uint8_t))));
    auto position = allocate<Velocity3D>(exec, "sph.geometry.input_position", n);
    auto mass = allocate<Values>(exec, "sph.geometry.input_mass", n);
    auto ids = allocate<decltype(fluid.ids)::non_const_type>(exec, "sph.geometry.input_ids", n);
    auto ghost = allocate<decltype(fluid.ghost)::non_const_type>(exec, "sph.geometry.input_ghost", n);
    Kokkos::parallel_for("sph.geometry.capture_fluid", Policy(exec, 0, nf), [&](std::size_t i) {
        ghost(i) = 0;
        ids(i) = fluid.ids(i);
        mass(i) = fluid.mass(i);
        for (int k = 0; k < 3; ++k)
            position(i, k) = fluid.position(i, k);
    });
    if (nw)
        capture_walls(position, mass, ids, ghost);
    exec.fence("sph.geometry.initial_inputs_ready");
    return {std::move(ticket), {position, mass, ids, ghost}};
}

} // namespace

WcsphStep3D::WcsphStep3D(const Kokkos::Serial& exec, const Budget& budget,
                         std::size_t max_cells, WcsphVerletParameters parameters)
    : exec_(exec), budget_(budget), max_cells_(max_cells), parameters_(parameters) {}

bool WcsphStep3D::release_position_buffer() noexcept {
    const bool released = trial_positions_.has_value();
    trial_positions_.reset();
    return released;
}

Result<WcsphStep3D::Ticket> WcsphStep3D::reserve_storage(std::size_t bytes) {
    auto reservation = budget_.reserve(bytes);
    if (!reservation.ok() && reservation.status().code() == StatusCode::resource_exhausted && release_position_buffer())
        return budget_.reserve(bytes);
    return reservation;
}

const SolidWallBoundary3D& WcsphStep3D::WallGroup::source() const {
    return std::visit([](const auto& group) -> const SolidWallBoundary3D& { return group; }, boundary);
}

const PrescribedMovingWall3D* WcsphStep3D::WallGroup::moving() const {
    return std::get_if<PrescribedMovingWall3D>(&boundary);
}

void WcsphStep3D::capture_groups(const std::vector<WallGroupInput3D>& inputs) {
    if (inputs.empty())
        return;
    const auto groups = require(detail::checked_array_bytes(inputs.size(), sizeof(WallGroup)));
    std::size_t bytes = wall_count_ * sizeof(WallParticle3D);
    const auto add_bytes = [&](std::size_t extra) {
        if (extra > std::numeric_limits<std::size_t>::max() - bytes)
            throw Failure{{StatusCode::resource_exhausted, "WCSPH wall group storage exceeds capacity"}};
        bytes += extra;
    };
    add_bytes(groups);
    for (const auto& input : inputs) {
        if (input.trajectory)
            add_bytes(input.trajectory->retained_bytes());
    }
    wall_storage_ = require(budget_.reserve(bytes));
    walls_.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (std::any_of(walls_.begin(), walls_.end(), [&](const auto& g) { return g.id == input.id; }))
            throw Failure{{StatusCode::invalid_argument, "duplicate WCSPH wall group ID"}};
        auto particles = require(SolidWallBoundary3D::capture_for_grid(
            exec_, input.particles, input.trajectory.has_value()));
        if (input.trajectory) {
            PrescribedMovingWall3D moving(particles, *input.trajectory);
            if (particles.extent(0) && moving.velocity_driven())
                has_velocity_driven_particles_ = true;
            walls_.push_back({input.id, moving_groups_++, std::move(moving)});
        } else {
            walls_.push_back({input.id, 0, FixedWall3D(std::move(particles))});
        }
    }
}

WallPhases3D::non_const_type WcsphStep3D::phase_storage(WcsphState3D& state) {
    if (!moving_groups_)
        return {};
    const auto bytes = require(detail::checked_array_bytes(moving_groups_, sizeof(WallPhase3D)));
    state.phase_storage = require(reserve_storage(bytes));
    return allocate<WallPhases3D::non_const_type>(exec_, "sph.wall.group_phases", moving_groups_);
}

WcsphStep3D::WallMotions WcsphStep3D::sample_wall_motion(WcsphState3D& candidate, WallMotionUse3D use,
                                                         const WcsphState3D* accepted, real dt) {
    WallMotions motions;
    if (!moving_groups_)
        return motions;
    const auto bytes = require(detail::checked_array_bytes(moving_groups_, sizeof(SampledWallMotion3D)));
    motions.ticket = require(reserve_storage(bytes));
    motions.values = decltype(motions.values)(
        Kokkos::view_alloc(exec_, "sph.wall.stage_motions"), moving_groups_);
    for (const auto& group : walls_) {
        const auto* moving = group.moving();
        if (!moving || !moving->particles().extent(0))
            continue;
        const auto index = group.phase_index;
        const auto* start_motion = accepted && moving->velocity_driven()
            ? &accepted->wall_motion(index) : nullptr;
        motions.values(index) = require(moving->motion_at_certified_time(
            candidate.wall_phases(index).time, use, start_motion, dt));
    }
    if (has_velocity_driven_particles_) {
        candidate.wall_motion_storage = motions.ticket;
        candidate.wall_motion = motions.values;
    }
    return motions;
}

void WcsphStep3D::set_next_step(WcsphState3D& candidate, real largest_speed_squared) const {
    candidate.maximum_step = require(central_difference_step::acoustic_step(largest_speed_squared,
        parameters_.eos.sound_speed, parameters_.cfl, parameters_.smoothing_length));
    for (std::size_t g = 0; g < candidate.wall_phases.extent(0); ++g)
        candidate.maximum_step = std::min(candidate.maximum_step, candidate.wall_phases(g).maximum_step);
}

std::optional<WcsphStep3D::MovingWallParticle> WcsphStep3D::moving_wall_particle(std::uint64_t particle_id) const {
    for (const auto& group : walls_) {
        if (const auto* moving = group.moving()) {
            if (const auto* particle = moving->find(particle_id))
                return MovingWallParticle{group.phase_index, particle->reference_position};
        }
    }
    return std::nullopt;
}

void WcsphStep3D::write_wall_positions(const ParticleFields3D& fields, const WallMotions& motions,
                                       const Velocity3D& output) const {
    if (!wall_count_)
        return;
    if (!moving_groups_) {
        Kokkos::parallel_for("sph.wall.fixed_stage", Policy(exec_, 0, fields.ids.extent(0)),
            [&](std::size_t slot) {
                if (fields.ghost(slot))
                    for (int k = 0; k < 3; ++k)
                        output(slot, k) = fields.position(slot, k);
            });
        return;
    }
    // Only drift needs all moving wall slots. Continuity samples their velocity on demand.
    // update_positions checks every drifted coordinate before converting it to an index.
    Kokkos::parallel_for("sph.wall.drift_position", Policy(exec_, 0, fields.ids.extent(0)),
        [&](std::size_t slot) {
            if (!fields.ghost(slot))
                return;
            if (const auto particle = moving_wall_particle(fields.ids(slot))) {
                const auto position = motions.values(particle->phase_index).position(particle->reference_position);
                for (int k = 0; k < 3; ++k)
                    output(slot, k) = position[k];
            } else {
                for (int k = 0; k < 3; ++k)
                    output(slot, k) = fields.position(slot, k);
            }
        });
}

WallTrajectoryTime3D WcsphStep3D::committed_wall_time(std::uint64_t particle_id, const WallPhases3D& phases) const {
    if (const auto particle = moving_wall_particle(particle_id))
        return phases(particle->phase_index).time;
    return {}; // A committed ghost outside the moving groups belongs to a fixed wall.
}

void WcsphStep3D::evaluate_pressure_acceleration(WcsphState3D& state, const CellGrid& grid,
    const BindFields3D& bind, const WallMotions& motions, bool initial_density) {
    const auto& q = grid.neighbors();
    const auto& sources = q.fields();
    const auto n = sources.ids.extent(0), rows = q.owned_slots().extent(0);
    const auto trim = [&] { return release_position_buffer(); };
    const auto write = [&](const StepOutputs3D& out) -> Status {
        Ticket scratch;
        if (rows) {
            const auto stride = step_pressure_workspace_bytes_per_source(n != rows);
            const auto bytes = n * stride;
            auto reserved = reserve_storage(bytes);
            if (!reserved.ok())
                return reserved.status();
            scratch = std::move(reserved).value();
        }
        const auto acceleration_at = [&](LocalIndex slot) {
            if (const auto particle = moving_wall_particle(sources.ids(slot)))
                return motions.values(particle->phase_index).acceleration(particle->reference_position);
            return std::array<real, 3>{};
        };
        const TaitEos eos(parameters_.eos);
        return evaluate_step_pressure_3d(exec_, q, eos, state.density->view(), out.pressure,
                                         out.acceleration, std::cref(acceleration_at), parameters_.gravity, initial_density);
    };
    require(bind(FieldGroup3D::pressure, write, trim));
}

Status WcsphStep3D::initialize_state(const ParticleFields3D& fluid,
                                     const Velocity3D::const_type& velocity,
                                     const Values::const_type& density,
                                     const std::vector<WallGroupInput3D>& initial_walls,
                                     WcsphState3D& candidate, const BindFields3D& bind) {
    return guarded([&] {
        require(parameters_.validate());
        const auto nf = fluid.position.extent(0);
        if (!valid_extent(nf, fluid.position, fluid.ids, fluid.mass, fluid.ghost, velocity, density))
            throw Failure{{StatusCode::invalid_argument, "WCSPH fluid field extents disagree"}};
        std::size_t remaining = std::numeric_limits<LocalIndex>::max();
        const auto admit_sources = [&](std::size_t count) {
            if (count > remaining)
                throw Failure{{StatusCode::resource_exhausted, "WCSPH source count exceeds capacity"}};
            remaining -= count;
        };
        admit_sources(nf);
        for (const auto& group : initial_walls)
            admit_sources(group.particles.ids.extent(0));
        wall_count_ = std::numeric_limits<LocalIndex>::max() - remaining - nf;
        std::size_t ghosts{};
        Kokkos::parallel_reduce("sph.wall.validate_ownership", Policy(exec_, 0, nf),
            [&](std::size_t i, std::size_t& count) { count += fluid.ghost(i) != 0; }, ghosts);
        if (ghosts)
            throw Failure{{StatusCode::unsupported, "WCSPH requires all-owned fluid input"}};
        capture_groups(initial_walls);
        const auto phases = phase_storage(candidate);
        for (const auto& group : walls_)
            if (const auto* moving = group.moving()) {
                const auto time = moving->initial_time();
                phases(group.phase_index) = {group.id, time, require(moving->initial_step_limit())};
            }
        candidate.wall_phases = phases;
        const auto samples = sample_wall_motion(candidate, nf ? WallMotionUse3D::full : WallMotionUse3D::position);
        const auto capture_walls = [&](const auto& position, const auto& mass,
                                       const auto& ids, const auto& ghost) {
            std::size_t offset = nf;
            for (const auto& group : walls_) {
                const auto& particles = group.source().particles();
                const auto count = particles.extent(0);
                const auto write = [&](std::size_t w, const std::array<real, 3>& x) {
                    const auto i = offset + w;
                    ghost(i) = 1;
                    ids(i) = particles(w).id;
                    mass(i) = particles(w).mass;
                    for (int k = 0; k < 3; ++k)
                        position(i, k) = x[k];
                };
                if (group.moving()) {
                    // Grid assembly validates the sampled coordinates before publication.
                    Kokkos::parallel_for("sph.geometry.capture_moving_walls", Policy(exec_, 0, count),
                        [&](std::size_t w) {
                            write(w, samples.values(group.phase_index).position(particles(w).reference_position));
                        });
                } else {
                    // The initial grid validates fixed coordinates and masses.
                    Kokkos::parallel_for("sph.geometry.capture_fixed_walls", Policy(exec_, 0, count),
                        [&](std::size_t w) { write(w, particles(w).reference_position); });
                }
                offset += count;
            }
        };
        {
            std::optional<InitialInput> captured;
            if (wall_count_)
                captured.emplace(capture_input(exec_, budget_, fluid, wall_count_, capture_walls));
            const auto admit = [&](CellGrid::StorageGroup, std::size_t bytes) { return budget_.reserve(bytes); };
            auto grid = require(CellGrid::build_initial(exec_, captured ? captured->fields : fluid,
                                                        nf, max_cells_, admit,
                                                        parameters_.smoothing_length));
            auto& geometry = candidate.geometry;
            if (grid.input_is_sorted()) {
                geometry.particle_storage = grid.field_storage();
                geometry.input = grid.fields();
            } else {
                if (!captured)
                    captured.emplace(capture_input(exec_, budget_, fluid, 0, capture_walls));
                geometry.particle_storage.fill(captured->ticket);
                geometry.input = captured->fields;
            }
            require(establish_owned_rows(exec_, geometry, grid.owned_slots(), grid.sorted_to_input(),
                                         [&](std::size_t bytes) { return budget_.reserve(bytes); }));
            committed_grid_ = std::make_unique<CellGrid>(std::move(grid));
        }
        stamp_geometry(candidate);
        const auto& map = candidate.geometry.owned_row_to_input;
        const auto rows = Policy(exec_, 0, map.extent(0));
        require(bind(FieldGroup3D::density,
            [&](const StepOutputs3D& out) -> Status {
                Kokkos::parallel_for("sph.continuity.initial_density", rows,
                    [&](std::size_t r) { out.density(r) = density(map(r)); });
                return {};
            }, {}));
        real largest_speed_squared{};
        require(bind(FieldGroup3D::velocity,
            [&](const StepOutputs3D& out) -> Status {
                const auto speed = central_difference_step::capture_velocity(exec_, map, velocity, out.velocity);
                if (speed.ok())
                    largest_speed_squared = speed.value();
                return speed.status();
            }, {}));
        evaluate_density_rate(candidate, *committed_grid_, samples);
        evaluate_pressure_acceleration(candidate, *committed_grid_, bind, samples, true);
        set_next_step(candidate, largest_speed_squared);
    });
}

WcsphStep3D::WallMotions WcsphStep3D::advance_wall_motion(const WcsphState3D& current,
                                                          double duration, WcsphState3D& candidate) {
    const auto phases = phase_storage(candidate);
    for (const auto& group : walls_) {
        if (const auto* moving = group.moving()) {
            const auto index = group.phase_index;
            const auto& phase = current.wall_phases(index);
            const auto [time, limit] = require(
                moving->advance_certified_time(phase.time, duration, phase.maximum_step));
            phases(index) = {phase.group_id, time, limit};
        }
    }
    candidate.wall_phases = phases;
    const bool has_fluid = current.geometry.owned_row_to_input.extent(0) != 0;
    return sample_wall_motion(candidate, has_fluid ? WallMotionUse3D::full : WallMotionUse3D::position,
                              &current, static_cast<real>(duration));
}

std::unique_ptr<CellGrid> WcsphStep3D::advance_positions_and_update_neighbors(
    const WcsphState3D& current, real dt, const WallMotions& wall_motion, WcsphState3D& candidate) {
    const auto admit_grid = [&](CellGrid::StorageGroup, std::size_t bytes) { return reserve_storage(bytes); };
    const auto fluid_count = current.geometry.owned_row_to_input.extent(0);
    std::optional<CellGrid> rebuilt;
    if (!committed_grid_)
        rebuilt.emplace(require(CellGrid::rebuild_committed(exec_, current.geometry.fields(),
            fluid_count, max_cells_, admit_grid, parameters_.smoothing_length)));
    const auto& old_grid = committed_grid_ ? *committed_grid_ : *rebuilt;
    const auto& sources = old_grid.fields();
    const auto count = sources.ids.extent(0);
    // Keep positions outside the evictable workspace while the grid consumes them.
    auto positions = std::exchange(trial_positions_, std::nullopt);
    if (!positions) {
        auto ticket = require(budget_.reserve(count * 3 * sizeof(real)));
        positions.emplace(std::move(ticket), allocate<Velocity3D>(exec_, "sph.central_difference.position", count));
    }
    central_difference_step::drift_positions(exec_, old_grid, current.velocity->view(),
        current.acceleration->view(), dt, positions->values, parameters_.gravity);
    write_wall_positions(sources, wall_motion, positions->values);
    auto updated = old_grid.update_positions(exec_, positions->values, max_cells_, admit_grid);
    if (updated.status().code() == StatusCode::invalid_argument)
        throw Failure{{StatusCode::numerical_failure, "invalid drifted WCSPH position"}};
    auto next_grid = std::make_unique<CellGrid>(require(std::move(updated)));
    auto& geometry = candidate.geometry;
    if (next_grid->input_is_sorted()) {
        geometry.particle_storage = next_grid->field_storage();
        geometry.input = next_grid->fields();
        trial_positions_ = std::move(positions);
    } else {
        geometry.particle_storage = {positions->ticket, old_grid.field_storage()[1], old_grid.field_storage()[2]};
        geometry.input = {positions->values, sources.mass, sources.ids, sources.ghost};
    }
    const bool layout_changed = !old_grid.input_is_sorted() || !next_grid->input_is_sorted();
    if (layout_changed) {
        require(establish_owned_rows(exec_, geometry, next_grid->owned_slots(), next_grid->sorted_to_input(),
                                     [&](std::size_t bytes) { return reserve_storage(bytes); }));
    } else {
        geometry.row_storage = current.geometry.row_storage;
        geometry.owned_row_to_input = current.geometry.owned_row_to_input;
    }
    geometry.topology = current.geometry.topology;
    geometry.geometry = next_grid->positions_changed_
        ? require(detail::checked_next_version(current.geometry.geometry)) : current.geometry.geometry;
    geometry.layout = layout_changed ? require(detail::checked_next_version(current.geometry.layout))
                                     : current.geometry.layout;
    stamp_geometry(candidate);

    // The grid input is in old sorted-slot order. Compose the two row maps
    // only when reordering could change the owned row for a stable particle ID.
    if (!next_grid->input_is_sorted() && fluid_count) {
        auto& history = candidate.history;
        if (wall_count_ == 0) {
            history.row_storage = geometry.row_storage;
            history.previous_row = geometry.owned_row_to_input;
        } else {
            const auto previous_rows = fluid_rows_by_slot(old_grid);
            history.row_storage = require(reserve_storage(fluid_count * sizeof(LocalIndex)));
            auto previous_row = allocate<Geometry3D::Indices::non_const_type>(
                exec_, "sph.central_difference.previous_row", fluid_count);
            Kokkos::parallel_for("sph.central_difference.match_history", Policy(exec_, 0, fluid_count),
                [&](std::size_t row) { previous_row(row) = previous_rows.values(geometry.owned_row_to_input(row)); });
            history.previous_row = previous_row;
        }
    }
    return next_grid;
}

real WcsphStep3D::advance_density_and_velocity(const WcsphState3D& current, double duration,
                                               WcsphState3D& candidate, const BindFields3D& bind) {
    using namespace central_difference_step;
    const auto& previous = current.history;
    auto& history = candidate.history;
    history.step_duration = duration;
    auto scheme = TimeScheme3D::startup;
    if (previous.step_duration > 0) {
        // After 39 centered advances, one AB2 step couples the two time levels again.
        if (duration == previous.step_duration && previous.centered_steps < 39) {
            scheme = TimeScheme3D::centered;
            history.centered_steps = previous.centered_steps + 1;
        } else {
            scheme = TimeScheme3D::adams_bashforth;
        }
    }
    const TimeWeights3D weights{scheme, static_cast<real>(duration), static_cast<real>(previous.step_duration)};
    const auto trim = [&] { return release_position_buffer(); };
    require(bind(FieldGroup3D::density, [&](const StepOutputs3D& out) {
        return advance_density(exec_, history.previous_row, previous.previous_row,
            current.density->view(), current.density_rate,
            previous.density ? previous.density->view() : Values::const_type{}, previous.density_rate,
            weights, out.density);
    }, trim));
    real largest_speed_squared{};
    require(bind(FieldGroup3D::velocity, [&](const StepOutputs3D& out) -> Status {
        auto speed = advance_velocity(exec_, history.previous_row, previous.previous_row,
            current.velocity->view(), current.acceleration->view(),
            previous.velocity ? previous.velocity->view() : Velocity3D::const_type{},
            previous.acceleration ? previous.acceleration->view() : Acceleration3D::const_type{},
            weights, out.velocity, parameters_.gravity);
        if (speed.ok())
            largest_speed_squared = speed.value();
        return speed.status();
    }, trim));
    return largest_speed_squared;
}

WcsphStep3D::Buffer<Geometry3D::Indices::non_const_type> WcsphStep3D::fluid_rows_by_slot(const CellGrid& grid) {
    const auto count = grid.fields().ids.extent(0);
    auto ticket = require(reserve_storage(count * sizeof(LocalIndex)));
    auto fluid_row = allocate<Geometry3D::Indices::non_const_type>(exec_, "sph.continuity.fluid_row", count);
    const auto& owned = grid.owned_slots();
    Kokkos::parallel_for("sph.continuity.map_fluid_rows", Policy(exec_, 0, owned.extent(0)),
        [&](std::size_t row) { fluid_row(owned(row)) = static_cast<LocalIndex>(row); });
    return {std::move(ticket), std::move(fluid_row)};
}

void WcsphStep3D::evaluate_density_rate(WcsphState3D& state, const CellGrid& grid,
                                        const WallMotions& wall_motion) {
    const auto rows = grid.owned_slots().extent(0);
    state.rate_storage = require(reserve_storage(rows * sizeof(real)));
    auto rate = allocate<DensityRate3D>(exec_, "sph.continuity.density_rate", rows);
    if (rows && wall_count_) {
        const auto fluid_row = fluid_rows_by_slot(grid);
        if (moving_groups_) {
            const auto count = grid.fields().ids.extent(0);
            const auto ticket = require(reserve_storage(count * (3 * sizeof(real) + sizeof(std::uint8_t))));
            const auto velocity = allocate<Velocity3D>(exec_, "sph.continuity.wall_velocity", count);
            const auto ready = allocate<StageVelocityReady3D>(exec_, "sph.continuity.wall_velocity_ready", count);
            const auto write_wall_velocity = [&](LocalIndex slot) {
                std::array<real, 3> wall_velocity{};
                if (const auto particle = moving_wall_particle(grid.fields().ids(slot))) {
                    const auto sampled = wall_motion.values(particle->phase_index).velocity(particle->reference_position);
                    if (!sampled)
                        return false;
                    wall_velocity = *sampled;
                }
                for (int axis = 0; axis < 3; ++axis)
                    velocity(slot, axis) = wall_velocity[axis];
                return true;
            };
            require(evaluate_density_rate_3d(exec_, grid.neighbors(), state.velocity->view(), fluid_row.values,
                                             rate, velocity, ready, std::cref(write_wall_velocity)));
        } else {
            require(evaluate_density_rate_3d(exec_, grid.neighbors(), state.velocity->view(), fluid_row.values, rate));
        }
    } else {
        require(evaluate_density_rate_3d(exec_, grid.neighbors(), state.velocity->view(), {}, rate));
    }
    state.density_rate = rate;
}

Status WcsphStep3D::try_step(const TimeInterval& interval, const WcsphState3D& current,
                             WcsphState3D& candidate, const BindFields3D& bind) {
    const double duration = interval.duration();
    const real dt = static_cast<real>(duration);
    if (!(dt > 0))
        return {StatusCode::invalid_argument, "WCSPH step rounds to zero in real precision"};

    const auto status = guarded([&] {
        // The accepted continuity rate and pressure force already include the wall at t_n.
        // Build the endpoint grid first to write fluid values in t_{n+1} owned-row order.
        const auto end_wall_motion = advance_wall_motion(current, duration, candidate);
        auto end_grid = advance_positions_and_update_neighbors(current, dt, end_wall_motion, candidate);
        const real largest_speed_squared = advance_density_and_velocity(current, duration, candidate, bind);
        evaluate_density_rate(candidate, *end_grid, end_wall_motion);
        evaluate_pressure_acceleration(candidate, *end_grid, bind, end_wall_motion);
        set_next_step(candidate, largest_speed_squared);
        candidate.history.density_rate = current.density_rate;
        committed_grid_ = std::move(end_grid);
    });
    if (!status.ok())
        release_position_buffer();
    return status;
}

void WcsphStep3D::release_workspace() noexcept {
    release_position_buffer();
    committed_grid_.reset();
}
} // namespace nexora::sph
