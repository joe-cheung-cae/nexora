#pragma once
#include <methods/sph/ParticleFields3D.hpp>
#include <participants/ResourceBudget.hpp>

#include <array>

namespace nexora::sph {

// Particle inputs retain the ordering supplied to this stage's grid. Rows map the
// four owned numerical fields into this input ordering, even after cache eviction.
struct Geometry3D {
    using Indices = Kokkos::View<const LocalIndex*, Kokkos::HostSpace>;
    using Ticket = detail::ResourceBudget::Ticket;
    std::array<Ticket, 3> particle_storage; // position, mass, ID/ghost
    Ticket row_storage;
    ParticleFields3D input;
    Indices owned_row_to_input;
    TopologyVersion topology{1};
    GeometryVersion geometry{1};
    LayoutVersion layout{1};

    const ParticleFields3D& fields() const { return input; }
};

Status establish_owned_rows(const Kokkos::Serial&, Geometry3D&, const Geometry3D::Indices& owned_slots,
                            const Geometry3D::Indices& sorted_to_input, const detail::ResourceBudget&);

} // namespace nexora::sph
