#pragma once

#include <nexora/core/Status.hpp>
#include <nexora/core/Real.hpp>

namespace nexora::sph {

struct WcsphParameters {
    real reference_density{};
    real sound_speed{};
    real gamma{};
    Status validate() const;
};

class TaitEos {
  public:
    explicit TaitEos(WcsphParameters parameters);
    Status validate() const;

    // The caller validates the input density and the representability of each result.
    real pressure(real density) const;
    real density_from_pressure(real pressure) const;

  private:
    WcsphParameters parameters_;
    real bulk_modulus_;
};

} // namespace nexora::sph
