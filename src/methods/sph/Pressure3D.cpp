#include "Pressure3D.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numbers>

namespace nexora::sph {
namespace {

using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;

template <class A, class B> bool overlaps(const A& a, const B& b) {
    if (!a.span() || !b.span())
        return false;
    const auto before = std::less<const real*>{};
    return before(a.data(), b.data() + b.span()) && before(b.data(), a.data() + a.span());
}

// Normalized binary64 significand with an extended exponent. This is used when
// an intermediate factor or partial sum cannot be kept in normal binary64.
struct Scaled {
    double significand{};
    int exponent{};

    static Scaled from(double value, int shift = 0) {
        if (value == 0)
            return {};
        int e;
        const double f = Kokkos::frexp(value, &e);
        return {f, e + shift};
    }

    Scaled times(double value) const {
        const auto factor = from(value);
        return from(significand * factor.significand, exponent + factor.exponent);
    }

    Scaled plus(Scaled other) const {
        if (significand == 0)
            return other;
        if (other.significand == 0)
            return *this;
        const int common = std::max(exponent, other.exponent);
        return from(Kokkos::ldexp(significand, exponent - common) +
                        Kokkos::ldexp(other.significand, other.exponent - common),
                    common);
    }

    double value() const {
        return Kokkos::ldexp(significand, exponent);
    }
};

bool fits_real(double value) {
    return Kokkos::isfinite(value) && Kokkos::abs(value) <= std::numeric_limits<real>::max();
}

// expm1 retains pressure perturbations near equilibrium. Use the logarithmic
// scale only if an intermediate exponent/scale overflows or underflows.
double tait_pressure(double rho, const WcsphParameters& p, double scale) {
    if (rho == p.reference_density)
        return 0;
    const double difference = rho - p.reference_density;
    const double ratio = rho / p.reference_density;
    double log_ratio;
    if (Kokkos::abs(difference) <= .5 * p.reference_density)
        log_ratio = Kokkos::log1p(difference / p.reference_density);
    else if (std::isnormal(ratio))
        log_ratio = Kokkos::log(ratio);
    else
        log_ratio = Kokkos::log(rho) - Kokkos::log(p.reference_density);
    const double exponent = p.gamma * log_ratio;
    const double delta = Kokkos::expm1(exponent);
    if (std::isnormal(scale) && std::isnormal(delta))
        return scale * delta;
    const double log_bulk = Kokkos::log(p.reference_density) + 2 * Kokkos::log(p.sound_speed);
    // expm1(z)/z tends to one, including when gamma*log_ratio rounds to zero.
    const double log_magnitude =
        Kokkos::abs(exponent) < std::numeric_limits<double>::epsilon()
            ? log_bulk + Kokkos::log(Kokkos::abs(log_ratio))
            : log_bulk - Kokkos::log(p.gamma) +
                  (Kokkos::isinf(delta) ? exponent : Kokkos::log(Kokkos::abs(delta)));
    return Kokkos::copysign(Kokkos::exp(log_magnitude), log_ratio);
}

// The scaled bulk modulus keeps the inverse defined even if rho0*c*c/gamma
// overflows or underflows on the way to a representable density.
double inverse_tait(double pressure, const WcsphParameters& p, Scaled scale) {
    if (pressure == 0)
        return p.reference_density;
    const auto parts = Scaled::from(pressure);
    const auto relative = Scaled::from(parts.significand / scale.significand,
                                       parts.exponent - scale.exponent);
    const double ratio = relative.value();
    double log_density_ratio;
    if (ratio < -.5) {
        // Subtract the unrounded numerator and denominator before dividing;
        // pressure / scale can round to -1 near the inverse domain boundary.
        const auto remainder = scale.plus(parts);
        if (remainder.significand <= 0)
            return std::numeric_limits<double>::quiet_NaN();
        log_density_ratio = (Kokkos::log(remainder.significand / scale.significand) +
                             (remainder.exponent - scale.exponent) * std::numbers::ln2) / p.gamma;
    } else if (Kokkos::abs(ratio) < std::numeric_limits<double>::epsilon()) {
        const auto gamma = Scaled::from(p.gamma);
        log_density_ratio = Scaled::from(relative.significand / gamma.significand,
                                          relative.exponent - gamma.exponent).value();
    } else if (Kokkos::isinf(ratio)) {
        log_density_ratio = (Kokkos::log(relative.significand) +
                             relative.exponent * std::numbers::ln2) / p.gamma;
    } else {
        log_density_ratio = Kokkos::log1p(ratio) / p.gamma;
    }
    const double multiplier = Kokkos::exp(log_density_ratio);
    if (std::isnormal(multiplier))
        return p.reference_density * multiplier;
    return Kokkos::exp(Kokkos::log(p.reference_density) + log_density_ratio);
}

double wall_pressure(const CellGrid& grid, LocalIndex wall, const std::array<double, 3>& gravity,
                     const Kokkos::View<LocalIndex*, Kokkos::HostSpace>& fluid_row,
                     const Values::const_type& density, const Values& pressure) {
    Scaled numerator{};
    double weights = 0;
    const auto& fields = grid.fields();
    grid.for_each_neighbor_of_slot(wall, [&](LocalIndex f, double d2) {
        if (fields.ghost(f))
            return;
        const double w = weight_3d(d2);
        if (w == 0)
            return;
        const auto row = fluid_row(f);
        Scaled head{};
        for (int axis = 0; axis < 3; ++axis) {
            const double dx = double(fields.position(wall, axis)) - fields.position(f, axis);
            head = head.plus(Scaled::from(gravity[axis]).times(dx));
        }
        const auto sample = head.times(density(row)).plus(Scaled::from(pressure(row)));
        numerator = numerator.plus(sample.times(w));
        weights += w;
    });
    if (weights <= 0)
        return std::numeric_limits<double>::quiet_NaN();
    return Kokkos::ldexp(numerator.significand / weights, numerator.exponent);
}

bool contributes(const CellGrid& grid, LocalIndex i, LocalIndex j, double d2) {
    if (d2 == 4)
        return false;
    if (d2 > 0)
        return true;
    // Two distinct subnormal coordinates may have a squared separation rounded to zero.
    const auto& x = grid.fields().position;
    return x(i, 0) != x(j, 0) || x(i, 1) != x(j, 1) || x(i, 2) != x(j, 2);
}

// -W'(q)/q with h=1; its first branch includes the finite q->0 limit.
double gradient_factor(double d2) {
    const double q = Kokkos::sqrt(d2);
    if (q < 1)
        return std::numbers::inv_pi * (3 - 2.25 * q);
    const double tail = 2 - q;
    return (.75 * std::numbers::inv_pi) * (tail * tail / q);
}

std::array<double, 3>
acceleration_at_slot(const CellGrid& grid, LocalIndex i,
                     const Kokkos::View<const double*, Kokkos::HostSpace>& force_coefficient,
                     const Kokkos::View<const int*, Kokkos::HostSpace>& force_exponent) {
    const auto& position = grid.fields().position;
    const auto& mass = grid.fields().mass;

    const auto each_pair = [&](auto visit) {
        grid.for_each_neighbor_of_slot(i, [&](LocalIndex j, double d2) {
            if (contributes(grid, i, j, d2))
                visit(j, gradient_factor(d2));
        });
    };
    const auto separation = [&](LocalIndex j, int axis) {
        return double(position(i, axis)) - double(position(j, axis));
    };
    std::array<double, 3> a{};
    bool scaled = force_exponent(i) != 0;
    if (!scaled)
        each_pair([&](LocalIndex j, double gradient) {
            if (scaled)
                return;
            if (force_exponent(j) != 0) {
                scaled = true;
                return;
            }
            const double sum = force_coefficient(i) + force_coefficient(j);
            if (sum == 0)
                return;
            const double weighted = sum * gradient;
            const double pair = weighted * double(mass(j));
            if (!std::isnormal(weighted) || !std::isnormal(pair)) {
                scaled = true;
                return;
            }
            for (int axis = 0; axis < 3; ++axis) {
                const double dx = separation(j, axis);
                if (dx == 0)
                    continue;
                const double term = pair * dx;
                a[axis] += term;
                if (!std::isnormal(term) || !Kokkos::isfinite(a[axis]))
                    scaled = true;
            }
        });
    if (scaled) {
        // Reevaluate the complete row in its original order. Term and accumulator
        // exponents stay separate until the final three components are narrowed.
        Scaled total[3]{};
        const auto source = [&](LocalIndex slot) {
            return Scaled::from(force_coefficient(slot), force_exponent(slot));
        };
        const auto self = source(i);
        each_pair([&](LocalIndex j, double gradient) {
            const auto pair = self.plus(source(j)).times(gradient).times(mass(j));
            for (int axis = 0; axis < 3; ++axis)
                total[axis] = total[axis].plus(pair.times(separation(j, axis)));
        });
        for (int axis = 0; axis < 3; ++axis)
            a[axis] = total[axis].value();
    }
    return a;
}

} // namespace

std::size_t pressure_workspace_bytes_per_source(bool has_ghosts, bool fixed_walls) {
    const auto per_ghost = sizeof(std::uint8_t) + (fixed_walls ? sizeof(LocalIndex) : 0);
    return sizeof(double) + sizeof(int) + (has_ghosts ? per_ghost : 0);
}

Status WcsphParameters::validate() const {
    if (!std::isfinite(reference_density) || reference_density <= 0 ||
        !std::isfinite(sound_speed) || sound_speed <= 0 || !std::isfinite(gamma) || gamma <= 0)
        return {StatusCode::invalid_argument,
                "WCSPH reference density, sound speed and gamma must be finite and positive"};
    return {};
}

Status evaluate_pressure_3d(const Kokkos::Serial& exec, const CellGrid& grid,
                            const WcsphParameters& parameters, const Values::const_type& density,
                            const Values& pressure, const Acceleration3D& acceleration,
                            const std::array<double, 3>* wall_gravity) {
    if (auto s = parameters.validate(); !s.ok())
        return s;
    if (wall_gravity && !std::all_of(wall_gravity->begin(), wall_gravity->end(),
                                     [](double x) { return std::isfinite(x); }))
        return {StatusCode::invalid_argument, "wall gravity must be finite"};
    const auto& rows = grid.owned_slots();
    const auto count = rows.extent(0);
    if (density.extent(0) != count || pressure.extent(0) != count ||
        acceleration.extent(0) != count ||
        (count && (!density.data() || !pressure.data() || !acceleration.data())) ||
        overlaps(density, pressure) || overlaps(density, acceleration) ||
        overlaps(pressure, acceleration))
        return {StatusCode::invalid_argument,
                "WCSPH requires three disjoint fields over the owned rows"};
    if (!count)
        return {};
    const auto n = grid.fields().ids.extent(0);
    const auto& ghost = grid.fields().ghost;
    // Split exponents once so an intermediate ratio cannot spoil a representable bulk scale.
    int er, ec, eg;
    const double r = Kokkos::frexp(parameters.reference_density, &er);
    const double c = Kokkos::frexp(parameters.sound_speed, &ec);
    const double g = Kokkos::frexp(parameters.gamma, &eg);
    const auto bulk_scale = Scaled::from((r / g) * c * c, er + 2 * ec - eg);
    const double scale = bulk_scale.value();
    try {
        Kokkos::View<double*, Kokkos::HostSpace> coefficient(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                               "sph.pressure_over_density_squared"),
            n);
        Kokkos::View<int*, Kokkos::HostSpace> exponent(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                               "sph.pressure.coefficient_exponent"),
            n);
        Kokkos::View<std::uint8_t*, Kokkos::HostSpace> needed;
        Kokkos::View<LocalIndex*, Kokkos::HostSpace> fluid_row;
        // A Serial pass marks ghosts participating in a nonzero gradient. Distant
        // and coincident ghosts need no pressure or reconstruction of their own density.
        const Policy owned(exec, 0, count);
        if (n != count) {
            if (wall_gravity)
                fluid_row = decltype(fluid_row)(
                    Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.wall.fluid_row"), n);
            needed = decltype(needed)(Kokkos::view_alloc(exec, "sph.pressure.needed_ghost"), n);
            Kokkos::parallel_for("sph.pressure.mark_ghosts", owned, [&](std::size_t row) {
                const auto i = rows(row);
                if (wall_gravity)
                    fluid_row(i) = static_cast<LocalIndex>(row);
                grid.for_each_neighbor_of_slot(i, [&](LocalIndex j, double d2) {
                    if (ghost(j) && contributes(grid, i, j, d2))
                        needed(j) = 1;
                });
            });
        }
        const auto store_coefficient = [&](LocalIndex slot, double rho, real p) {
            // Keep ordinary coefficients directly readable in the force hot path.
            // Preserve tiny/huge quotients in scaled form until the mass and
            // displacement factors are included.
            const double first = double(p) / rho;
            const double ratio = first / rho;
            if (p == 0 || (std::isnormal(first) && std::isnormal(ratio))) {
                coefficient(slot) = ratio;
                exponent(slot) = 0;
            } else {
                const auto pressure_parts = Scaled::from(p);
                const auto density_parts = Scaled::from(rho);
                const auto scaled =
                    Scaled::from(pressure_parts.significand /
                                     (density_parts.significand * density_parts.significand),
                                 pressure_parts.exponent - 2 * density_parts.exponent);
                coefficient(slot) = scaled.significand;
                exponent(slot) = scaled.exponent;
            }
        };
        const auto prepare_source = [&](LocalIndex slot, double rho, real& p) {
            if (!Kokkos::isfinite(rho) || rho <= 0)
                return false;
            const double raw = tait_pressure(rho, parameters, scale);
            if (!fits_real(raw))
                return false;
            p = static_cast<real>(raw);
            store_coefficient(slot, rho, p);
            return true;
        };
        std::size_t invalid{};
        Kokkos::parallel_reduce(
            "sph.pressure.owned_eos", owned,
            [&](std::size_t row, std::size_t& errors) {
                if (!prepare_source(rows(row), density(row), pressure(row)))
                    ++errors;
            },
            invalid);
        if (invalid)
            return {StatusCode::numerical_failure,
                    "invalid owned WCSPH density or pressure exceeds real range"};
        if (n != count) {
            Kokkos::parallel_reduce(
                "sph.pressure.ghost_eos", Policy(exec, 0, n),
                [&](std::size_t slot, std::size_t& errors) {
                    if (!needed(slot))
                        return;
                    const auto j = static_cast<LocalIndex>(slot);
                    if (wall_gravity) {
                        const double raw = wall_pressure(grid, j, *wall_gravity, fluid_row,
                                                         density, pressure);
                        if (!fits_real(raw)) {
                            ++errors;
                            return;
                        }
                        const real p = static_cast<real>(raw);
                        const double rho = inverse_tait(p, parameters, bulk_scale);
                        if (!fits_real(rho) || static_cast<real>(rho) <= 0) {
                            ++errors;
                            return;
                        }
                        store_coefficient(j, static_cast<real>(rho), p);
                        return;
                    }
                    const double rho = density_at_slot_3d(grid, j);
                    real p{};
                    if (!fits_real(rho) || !prepare_source(j, static_cast<real>(rho), p))
                        ++errors;
                },
                invalid);
            if (invalid)
                return {StatusCode::numerical_failure,
                        wall_gravity ? "invalid Adami wall pressure or inverse Tait density"
                                     : "invalid reconstructed ghost WCSPH density or pressure exceeds real range"};
        }
        const decltype(coefficient)::const_type force_coefficient = coefficient;
        const decltype(exponent)::const_type force_exponent = exponent;
        Kokkos::parallel_reduce(
            "sph.pressure.acceleration", owned,
            [&](std::size_t row, std::size_t& errors) {
                const auto a =
                    acceleration_at_slot(grid, rows(row), force_coefficient, force_exponent);
                for (int axis = 0; axis < 3; axis++) {
                    if (fits_real(a[axis]))
                        acceleration(row, axis) = static_cast<real>(a[axis]);
                    else
                        ++errors;
                }
            },
            invalid);
        if (invalid)
            return {StatusCode::numerical_failure,
                    "WCSPH pressure acceleration exceeds real range"};
        return {};
    } catch (const std::bad_alloc&) {
        return {StatusCode::resource_exhausted, "cannot allocate WCSPH pressure scratch"};
    } catch (const std::exception&) {
        return {StatusCode::execution_failure, "cannot evaluate WCSPH pressure acceleration"};
    }
}
} // namespace nexora::sph
