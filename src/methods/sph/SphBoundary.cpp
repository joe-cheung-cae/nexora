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

FixedWall3D::FixedWall3D(ParticleView sorted_particles)
    : SolidWallBoundary3D(std::move(sorted_particles)) {}

Result<SolidWallBoundary3D::ParticleView>
SolidWallBoundary3D::capture_for_grid(const Kokkos::Serial& exec, const FixedWallParticles3D& input,
                                     bool moving) {
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
    if (moving) {
        std::size_t invalid{};
        Kokkos::parallel_reduce(
            "sph.wall.capture_particles", Policy(exec, 0, n),
            [&](std::size_t i, std::size_t& errors) {
                const auto record = record_at(i);
                if (!Kokkos::isfinite(record.reference_position[0]) ||
                    !Kokkos::isfinite(record.reference_position[1]) ||
                    !Kokkos::isfinite(record.reference_position[2]))
                    ++errors;
                records(i) = record;
            }, invalid);
        if (invalid)
            return Status{StatusCode::invalid_argument, "invalid SPH moving wall reference position"};
    } else {
        Kokkos::parallel_for("sph.wall.capture_particles", Policy(exec, 0, n),
                             [&](std::size_t i) { records(i) = record_at(i); });
    }
    if (n > 1) {
        Kokkos::sort(exec, records, [](const WallParticle3D& a, const WallParticle3D& b) {
            return a.id < b.id;
        });
    }
    return SolidWallBoundary3D::ParticleView(records);
}

} // namespace nexora::sph::SphBoundary
