#include "TaitEos.hpp"

#include <Kokkos_MathematicalFunctions.hpp>
#include <cmath>

namespace nexora::sph {

Status WcsphParameters::validate() const {
    if (!std::isfinite(reference_density) || reference_density <= 0 ||
        !std::isfinite(sound_speed) || sound_speed <= 0 || !std::isfinite(gamma) || gamma <= 0)
        return {StatusCode::invalid_argument,
                "WCSPH reference density, sound speed and gamma must be finite and positive"};
    return {};
}

TaitEos::TaitEos(WcsphParameters parameters)
    : parameters_(parameters), bulk_modulus_(parameters.reference_density * parameters.sound_speed *
                                           parameters.sound_speed / parameters.gamma) {}

Status TaitEos::validate() const {
    return parameters_.validate();
}

real TaitEos::pressure(real density) const {
    // log1p and expm1 retain small pressure perturbations near equilibrium.
    const real reference = parameters_.reference_density;
    const real difference = density - reference;
    const real log_ratio = Kokkos::abs(difference) <= real{.5} * reference
                                 ? Kokkos::log1p(difference / reference)
                                 : Kokkos::log(density / reference);
    return bulk_modulus_ * Kokkos::expm1(parameters_.gamma * log_ratio);
}

real TaitEos::density_from_pressure(real pressure) const {
    return parameters_.reference_density *
           Kokkos::exp(Kokkos::log1p(pressure / bulk_modulus_) / parameters_.gamma);
}

} // namespace nexora::sph
