#include <nexora/data/FieldRegistry.hpp>
#include <nexora/data/FieldStamp.hpp>

int main() {
    nexora::FieldDescriptor field;
    field.name = "pressure";
    field.units = "Pa";
    const nexora::FieldStamp stamp;
    nexora::FieldRegistry registry;
    return field.scalar_type == nexora::real_scalar_type && stamp.validate().ok() &&
                   registry.add(field).ok() && registry.find(field.id)
               ? 0
               : 1;
}
