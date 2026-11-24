#include "Continuity3D.hpp"

#include <functional>

namespace nexora::sph {
namespace {
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
template <class A, class B> bool overlaps(const A& a, const B& b) {
    if (!a.span() || !b.span())
        return false;
    const auto before = std::less<const void*>{};
    return before(a.data(), b.data() + b.span()) && before(b.data(), a.data() + a.span());
}

template <class... Views> bool valid_fields(std::size_t n, const Views&... views) {
    return ((views.extent(0) == n && (!n || views.data() != nullptr)) && ...);
}

} // namespace

Status continuity_rate_3d(const Kokkos::Serial& exec, const NeighborView3D& grid,
                          const Velocity3D::const_type& velocity, const DensityRate3D& rate) {
    const auto& sources = grid.fields();
    const auto owned = grid.owned_slots();
    if (!valid_fields(sources.ids.extent(0), velocity) || !valid_fields(owned.extent(0), rate) ||
        overlaps(rate, velocity) || overlaps(rate, sources.mass) || overlaps(rate, sources.position))
        return {StatusCode::invalid_argument, "continuity fields have invalid extents, storage or output overlap"};
    const double h = grid.smoothing_length();
    const double inverse_h5 = 1 / (h * h * h * h * h);
    std::size_t invalid{};
    Kokkos::parallel_reduce(
        "sph.continuity.rate", Policy(exec, 0, owned.extent(0)),
        [&](std::size_t row, std::size_t& errors) {
            const auto i = owned(row);
            double sum = 0;
            grid.for_each_neighbor(row, [&](LocalIndex j, double d2) {
                const double gradient =
                    m4_negative_radial_derivative_over_q_3d(Kokkos::sqrt(d2) / h);
                if (gradient == 0)
                    return;
                double dot = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    const double vj = velocity(j, axis);
                    const double dx = double(sources.position(j, axis)) - sources.position(i, axis);
                    dot += (double(velocity(i, axis)) - vj) * dx;
                }
                sum += (dot * sources.mass(j)) * gradient * inverse_h5;
            });
            rate(row) = sum;
            errors += !Kokkos::isfinite(sum);
        }, invalid);
    if (invalid)
        return {StatusCode::numerical_failure, "nonfinite SPH continuity rate"};
    return {};
}

void correct_density_3d(const Kokkos::Serial& exec, const NeighborView3D::Indices& new_owned,
                        const NeighborView3D::Indices& new_to_old_slot,
                        const NeighborView3D::Indices& old_row, const Values::const_type& density,
                        const DensityRate3D& rate_n, const DensityRate3D& rate_star,
                        double dt, const Values& corrected) {
    Kokkos::parallel_for("sph.continuity.correct_density", Policy(exec, 0, new_owned.extent(0)),
        [&](std::size_t r) {
            const auto r0 = old_row(new_to_old_slot(new_owned(r)));
            corrected(r) = narrow_step_value(Kokkos::fma(
                .5 * dt, rate_n(r0) + rate_star(r), double(density(r0))));
        });
}

} // namespace nexora::sph
