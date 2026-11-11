#pragma once
#include <cstddef>
#include <nexora/core/Status.hpp>
#include <nexora/core/Types.hpp>
#include <string>
#include <type_traits>
namespace nexora {
enum class Association {
    cell,
    face,
    node,
    fe_dof,
    quadrature_point,
    particle,
    contact,
    lattice_population,
    boundary_sample,
    rigid_body,
    joint,
    generalized_position,
    generalized_velocity,
    constraint,
    modal_coordinate
};
enum class ScalarType { float32, float64, int32, int64, uint32, uint64 };
inline constexpr ScalarType real_scalar_type =
    std::is_same_v<real, float> ? ScalarType::float32 : ScalarType::float64;
enum class FieldOwnership { owned, ghost, replicated, external };
enum class FieldAccess { read, overwrite, accumulate };
enum class MeasureSemantics { point_value, density, integrated_load, flux_integral };
struct FieldDescriptor {
    FieldId id{};
    std::string name;
    DomainId domain_id{};
    Association association{Association::cell};
    std::string entity_set;
    ScalarType scalar_type{real_scalar_type};
    std::size_t components{1};
    std::string tensor_interpretation;
    std::string units;
    std::string reference_frame;
    std::string pressure_convention;
    std::string time_level;
    std::string stage;
    FieldOwnership ownership{FieldOwnership::owned};
    FieldAccess access{FieldAccess::read}; // Maximum declared access, not a lease.
    MeasureSemantics measure{MeasureSemantics::point_value};
    Status validate() const {
        if (name.empty())
            return {StatusCode::invalid_argument, "field name is empty"};
        if (units.empty())
            return {StatusCode::invalid_argument, "field units are empty"};
        if (!components)
            return {StatusCode::invalid_argument, "field component count is zero"};
        if (static_cast<unsigned>(association) >
            static_cast<unsigned>(Association::modal_coordinate))
            return {StatusCode::invalid_argument, "invalid field association"};
        if (static_cast<unsigned>(scalar_type) > static_cast<unsigned>(ScalarType::uint64))
            return {StatusCode::invalid_argument, "invalid field scalar type"};
        if (static_cast<unsigned>(ownership) > static_cast<unsigned>(FieldOwnership::external))
            return {StatusCode::invalid_argument, "invalid field ownership"};
        if (static_cast<unsigned>(access) > static_cast<unsigned>(FieldAccess::accumulate))
            return {StatusCode::invalid_argument, "invalid field access"};
        if (static_cast<unsigned>(measure) > static_cast<unsigned>(MeasureSemantics::flux_integral))
            return {StatusCode::invalid_argument, "invalid field measure"};
        return {};
    }
};
} // namespace nexora
