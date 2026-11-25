#include "Geometry3D.hpp"

#include <data/EntityCapacity.hpp>

namespace nexora::sph {

Status establish_owned_rows(const Kokkos::Serial& exec, Geometry3D& geometry,
                            const Geometry3D::Indices& owned_slots,
                            const Geometry3D::Indices& sorted_to_input,
                            const std::function<Result<Geometry3D::Ticket>(std::size_t)>& reserve) {
    const auto n = owned_slots.extent(0);
    const auto bytes = detail::checked_array_bytes(n, sizeof(LocalIndex));
    if (!bytes.ok())
        return bytes.status();
    auto ticket = reserve(bytes.value());
    if (!ticket.ok())
        return ticket.status();
    Geometry3D::Indices::non_const_type rows(
        Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.geometry.owned_row_to_input"), n);
    using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for("sph.geometry.map_owned_rows", Policy(exec, 0, n), [&](std::size_t row) {
        rows(row) = sorted_to_input(owned_slots(row));
    });
    exec.fence("sph.geometry.owned_rows_ready");
    geometry.row_storage = std::move(ticket).value();
    geometry.owned_row_to_input = rows;
    return {};
}

} // namespace nexora::sph
