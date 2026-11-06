#pragma once

#include <nexora/core/Types.hpp>
#include <nexora/data/FieldDescriptor.hpp>

namespace nexora {

enum class PortDirection { input, output };

// Lightweight declaration only: field metadata carries the physical semantics.
// A valid declaration does not establish compatibility with another port.
struct PortDescriptor {
    PortId id;
    SupportId support;
    PortDirection direction = PortDirection::output;
    FieldDescriptor field;

    Status validate() const {
        if (direction != PortDirection::input && direction != PortDirection::output) {
            return {StatusCode::invalid_argument, "invalid participant port direction"};
        }
        return field.validate();
    }
};

} // namespace nexora
