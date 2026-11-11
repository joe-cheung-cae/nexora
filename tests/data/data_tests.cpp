#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <limits>
#include <nexora/data/FieldRegistry.hpp>
#include <nexora/data/FieldStamp.hpp>
int main() {
    using namespace nexora;
    FieldRegistry registry;
    FieldDescriptor field;
    assert(field.scalar_type == real_scalar_type);
    field.id = FieldId{2};
    assert(!registry.add(field).ok());
    field.name = "pressure";
    assert(!registry.add(field).ok());
    field.units = "Pa";
    field.components = 0;
    assert(!registry.add(field).ok());
    field.components = 1;
    field.association = static_cast<Association>(999);
    assert(!registry.add(field).ok());
    field.association = Association::cell;
    field.scalar_type = static_cast<ScalarType>(999);
    assert(!registry.add(field).ok());
    field.scalar_type = real_scalar_type;
    field.ownership = static_cast<FieldOwnership>(999);
    assert(!registry.add(field).ok());
    field.ownership = FieldOwnership::owned;
    field.access = static_cast<FieldAccess>(999);
    assert(!registry.add(field).ok());
    field.access = FieldAccess::read;
    field.measure = static_cast<MeasureSemantics>(999);
    assert(!registry.add(field).ok());
    field.measure = MeasureSemantics::point_value;
    assert(registry.add(field).ok());
    field.units = "bar";
    assert(!registry.add(field).ok());
    assert(registry.size() == 1);
    assert(registry.find(FieldId{3}) == nullptr);
    assert(registry.find(FieldId{2})->units == "Pa");

    FieldStamp stamp;
    assert(stamp.validate().ok());
    stamp.valid_from = 2;
    stamp.valid_until = 1;
    assert(!stamp.validate().ok());
    stamp.valid_until = 3;
    assert(!stamp.validate().ok());
    stamp.time.physical_time = 3;
    assert(stamp.validate().ok());
    stamp.time.physical_time = std::numeric_limits<double>::quiet_NaN();
    assert(!stamp.validate().ok());
    stamp.time.physical_time = 2;
    stamp.valid_until = std::numeric_limits<double>::infinity();
    assert(!stamp.validate().ok());
}
