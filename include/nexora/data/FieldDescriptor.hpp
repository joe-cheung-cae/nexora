#pragma once
#include <cmath>
#include <cstddef>
#include <nexora/core/Status.hpp>
#include <nexora/core/Types.hpp>
#include <nexora/exec/MemorySpace.hpp>
#include <string>
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
enum class FieldOwnership { owned, ghost, replicated, external };
enum class FieldAccess { read, overwrite, accumulate };
enum class MeasureSemantics { point_value, density, integrated_load, flux_integral };
struct FieldDescriptor {
    FieldId id{};
    std::string name;
    DomainId domain_id{};
    Association association{Association::cell};
    std::string entity_set;
    ScalarType scalar_type{ScalarType::float64};
    std::size_t components{1};
    std::string tensor_interpretation;
    std::string units;
    std::string reference_frame;
    std::string pressure_convention;
    std::string time_level;
    std::string stage;
    double valid_from{};
    double valid_until{};
    std::string layout;
    std::size_t entity_stride{};
    std::size_t component_stride{};
    MemorySpace memory_space{MemorySpace::host};
    int device_id{};
    FieldOwnership ownership{FieldOwnership::owned};
    TopologyVersion topology_version{};
    GeometryVersion geometry_version{};
    LayoutVersion layout_version{};
    StateVersion state_version{};
    FieldAccess access{FieldAccess::read};
    MeasureSemantics measure{MeasureSemantics::point_value};
    Status validate() const {
        if (name.empty())
            return {StatusCode::invalid_argument, "field name is empty"};
        if (units.empty())
            return {StatusCode::invalid_argument, "field units are empty"};
        if (!components)
            return {StatusCode::invalid_argument, "field component count is zero"};
        if (!std::isfinite(valid_from) || !std::isfinite(valid_until) || valid_until < valid_from)
            return {StatusCode::invalid_argument, "invalid validity interval"};
        if (static_cast<unsigned>(association) >
            static_cast<unsigned>(Association::modal_coordinate))
            return {StatusCode::invalid_argument, "invalid field association"};
        return {};
    }
};
} // namespace nexora
