#pragma once

#include <Kokkos_Core.hpp>
#include <array>
#include <data/CsrAdjacency.hpp>
#include <data/FieldIssuer.hpp>
#include <memory>
#include <nexora/core/Real.hpp>
#include <span>
#include <vector>

namespace nexora::reference {
using StructureValues = Kokkos::View<real*, Kokkos::HostSpace>;
using ConstStructureValues = StructureValues::const_type;
using StructureIds = detail::EntityOrder::id_view_type;

// This internal concrete state owns one nonempty, locally owned entity domain.
// There is no ghost/owner payload or numerical evolution. x is geometry, q is
// a scalar field, and history is passive per-entity state. CSR depends on the
// topology and both entity layouts; reorder preserves row-local edge order.
struct StructureConfig {
    detail::EntityNamespace entity_namespace;
    std::vector<GlobalEntityId> ids;
    std::vector<real> x, q, history;
    std::vector<Offset> offsets;
    std::vector<LocalIndex> indices;
    // Stable IDs, geometry x, scalar q, passive history, respectively.
    std::array<FieldDescriptor, 4> fields;
    FieldStamp stamp;
};

template <class View> struct StructureField {
    View values;
    detail::FieldIssuer issuer;
    ReadField<View> grant;
};

struct StructureBank {
    // Banks are assembled by StructureState from an admitted logical stamp.
    class ConstructionKey {
        friend class StructureState;
        ConstructionKey() = default;
    };
    detail::EntityOrder order;
    detail::CsrAdjacency adjacency;
    FieldStamp logical;
    StructureField<StructureIds> ids;
    StructureField<ConstStructureValues> x, q, history;

    StructureBank(ConstructionKey, detail::EntityOrder, detail::CsrAdjacency, FieldStamp,
                  const std::array<FieldDescriptor, 4>&, ConstStructureValues,
                  ConstStructureValues, ConstStructureValues);
    void close() noexcept;
};

// Single-caller synchronous operations publish one fully prepared immutable
// bank. Existing snapshots retain their grants; fresh old bindings are closed.
// The owner and all snapshots must be destroyed before Kokkos finalization.
class StructureState {
  public:
    enum class Fault { none, after_partial_candidate, before_publish };
    static Result<std::unique_ptr<StructureState>> create(const Kokkos::Serial&,
                                                         const StructureConfig&);
    ~StructureState();
    StructureState(const StructureState&) = delete;
    StructureState& operator=(const StructureState&) = delete;
    std::shared_ptr<const StructureBank> snapshot() const;

    // Only layout and allocation incarnations change in a pure reorder.
    Status reorder(std::span<const GlobalEntityId>, Fault = Fault::none);
    // Explicit supplied values, not a solver or a window lifecycle.
    Status commit_values(std::span<const real>);

  private:
    StructureState(const Kokkos::Serial&, std::shared_ptr<StructureBank>);
    void publish(std::shared_ptr<StructureBank>) noexcept;
    Kokkos::Serial exec_;
    std::shared_ptr<StructureBank> accepted_;
};
} // namespace nexora::reference
