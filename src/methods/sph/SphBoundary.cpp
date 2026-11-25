#include "SphBoundary.hpp"

#include <Kokkos_Sort.hpp>

#include <algorithm>
#include <utility>

namespace nexora::sph::SphBoundary {

SolidWallBoundary3D::SolidWallBoundary3D(ParticleView sorted_particles)
    : particles_(std::move(sorted_particles)) {}

const WallParticle3D* SolidWallBoundary3D::find(std::uint64_t id) const {
    if (particles_.extent(0) == 0)
        return nullptr;
    const auto* begin = particles_.data();
    const auto* end = begin + particles_.extent(0);
    const auto* particle = std::lower_bound(begin, end, id, [](const auto& record, auto key) {
        return record.id < key;
    });
    return particle != end && particle->id == id ? particle : nullptr;
}

Result<WallKinematics3D> SolidWallBoundary3D::sample(std::uint64_t id, WallTrajectoryTime3D time) const {
    const auto* particle = find(id);
    if (!particle)
        return Status{StatusCode::invalid_argument, "unknown SPH wall particle ID"};
    return sample_reference(particle->reference_position, time);
}

FixedWall3D::FixedWall3D(ParticleView sorted_particles)
    : SolidWallBoundary3D(std::move(sorted_particles)) {}

Result<WallKinematics3D>
FixedWall3D::sample_reference(const std::array<real, 3>& position, WallTrajectoryTime3D) const {
    return kinematics(position);
}

namespace {
Result<SolidWallBoundary3D::ParticleView>
capture_records(const Kokkos::Serial& exec, const FixedWallParticles3D& input,
                bool independent, bool moving) {
    const auto n = input.ids.extent(0);
    if (input.position.extent(0) != n || input.mass.extent(0) != n ||
        (n && (!input.position.data() || !input.mass.data() || !input.ids.data())))
        return Status{StatusCode::invalid_argument, "invalid SPH wall particle fields"};
    if (!n)
        return SolidWallBoundary3D::ParticleView{};
    SolidWallBoundary3D::ParticleView::non_const_type records(
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.wall.particles"), n);
    using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
    const auto record_at = [&](std::size_t i) {
        return WallParticle3D{input.ids(i),
                              {input.position(i, 0), input.position(i, 1), input.position(i, 2)},
                              input.mass(i)};
    };
    std::size_t invalid{};
    if (independent || moving) {
        Kokkos::parallel_reduce(
            "sph.wall.capture_particles", Policy(exec, 0, n),
            [&](std::size_t i, std::size_t& errors) {
                const auto record = record_at(i);
                if ((independent && (!Kokkos::isfinite(record.mass) || record.mass <= 0)) ||
                    !Kokkos::isfinite(record.reference_position[0]) ||
                    !Kokkos::isfinite(record.reference_position[1]) ||
                    !Kokkos::isfinite(record.reference_position[2]))
                    ++errors;
                records(i) = record;
            }, invalid);
        if (invalid)
            return Status{StatusCode::invalid_argument, "invalid SPH wall position or mass"};
    } else {
        Kokkos::parallel_for("sph.wall.capture_particles", Policy(exec, 0, n),
                             [&](std::size_t i) { records(i) = record_at(i); });
    }
    if (n > 1) {
        Kokkos::sort(exec, records, [](const WallParticle3D& a, const WallParticle3D& b) {
            return a.id < b.id;
        });
        if (independent) {
            Kokkos::parallel_reduce(
                "sph.wall.unique_ids", Policy(exec, 1, n),
                [&](std::size_t i, std::size_t& errors) { errors += records(i - 1).id == records(i).id; },
                invalid);
            if (invalid)
                return Status{StatusCode::invalid_argument, "SPH wall particle IDs must be unique"};
        }
    }
    return SolidWallBoundary3D::ParticleView(records);
}
} // namespace

Result<SolidWallBoundary3D::ParticleView>
capture_wall_particles(const Kokkos::Serial& exec, const FixedWallParticles3D& input) {
    return capture_records(exec, input, true, false);
}

Result<SolidWallBoundary3D::ParticleView>
SolidWallBoundary3D::capture_for_grid(const Kokkos::Serial& exec, const FixedWallParticles3D& input,
                                     bool moving) {
    return capture_records(exec, input, false, moving);
}

} // namespace nexora::sph::SphBoundary
