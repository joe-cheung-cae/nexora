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

Result<WallKinematics3D> SolidWallBoundary3D::sample(std::uint64_t id, double phase) const {
    const auto* particle = find(id);
    if (!particle)
        return Status{StatusCode::invalid_argument, "unknown SPH wall particle ID"};
    return sample_reference(particle->reference_position, phase);
}

FixedWall3D::FixedWall3D(ParticleView sorted_particles)
    : SolidWallBoundary3D(std::move(sorted_particles)) {}

Result<WallKinematics3D>
FixedWall3D::sample_reference(const std::array<real, 3>& position, double) const {
    return WallKinematics3D{position, {}, {}};
}

Result<SolidWallBoundary3D::ParticleView>
capture_wall_particles(const Kokkos::Serial& exec, const FixedWallParticles3D& input) {
    const auto n = input.ids.extent(0);
    if (input.position.extent(0) != n || input.mass.extent(0) != n ||
        (n && (!input.position.data() || !input.mass.data() || !input.ids.data())))
        return Status{StatusCode::invalid_argument, "invalid SPH wall particle fields"};
    if (!n)
        return SolidWallBoundary3D::ParticleView{};
    SolidWallBoundary3D::ParticleView::non_const_type records(
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.wall.particles"), n);
    using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
    std::size_t invalid{};
    Kokkos::parallel_reduce(
        "sph.wall.capture_particles", Policy(exec, 0, n),
        [&](std::size_t i, std::size_t& errors) {
            const std::array position{input.position(i, 0), input.position(i, 1),
                                      input.position(i, 2)};
            const real mass = input.mass(i);
            if (!Kokkos::isfinite(mass) || mass <= 0 || !Kokkos::isfinite(position[0]) ||
                !Kokkos::isfinite(position[1]) || !Kokkos::isfinite(position[2]))
                ++errors;
            records(i) = {input.ids(i), position, mass};
        },
        invalid);
    if (invalid)
        return Status{StatusCode::invalid_argument, "invalid SPH wall position or mass"};
    if (n > 1) {
        Kokkos::sort(exec, records, [](const WallParticle3D& a, const WallParticle3D& b) {
            return a.id < b.id;
        });
        Kokkos::parallel_reduce(
            "sph.wall.unique_ids", Policy(exec, 1, n),
            [&](std::size_t i, std::size_t& errors) { errors += records(i - 1).id == records(i).id; },
            invalid);
        if (invalid)
            return Status{StatusCode::invalid_argument, "SPH wall particle IDs must be unique"};
    }
    return SolidWallBoundary3D::ParticleView(records);
}

} // namespace nexora::sph::SphBoundary
