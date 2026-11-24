#pragma once

#include <nexora/core/Status.hpp>

namespace nexora::sph {

struct WcsphParameters {
    double reference_density{};
    double sound_speed{};
    double gamma{};
    Status validate() const;
};

class TaitEos {
  public:
    explicit TaitEos(WcsphParameters parameters);
    Status validate() const;

    // The caller validates the input density and the representability of each result.
    double pressure(double density) const;
    double density_from_pressure(double pressure) const;

  private:
    WcsphParameters parameters_;
    double bulk_modulus_;
};

} // namespace nexora::sph
