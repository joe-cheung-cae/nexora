#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <nexora/data/FieldRegistry.hpp>
#include <nexora/data/FieldView.hpp>
int main() {
    using namespace nexora;
    int storage[4]{};
    FieldView<int> view({storage, 4}, 2);
    assert(view.entities() == 2 && view.components() == 2);
    FieldView<int> empty({}, 2);
    assert(empty.entities() == 0);
    bool threw = false;
    try {
        FieldView<int> bad({storage, 4}, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    threw = false;
    try {
        FieldView<int> bad({storage, 4}, 3);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    FieldRegistry registry;
    FieldDescriptor field;
    field.id = FieldId{2};
    assert(!registry.add(field).ok());
    field.name = "pressure";
    assert(!registry.add(field).ok());
    field.units = "Pa";
    field.components = 0;
    assert(!registry.add(field).ok());
    field.components = 1;
    field.valid_from = 2;
    field.valid_until = 1;
    assert(!registry.add(field).ok());
    field.valid_until = 3;
    field.association = static_cast<Association>(999);
    assert(!registry.add(field).ok());
    field.association = Association::cell;
    field.layout_version = LayoutVersion{7};
    assert(registry.add(field).ok());
    assert(!registry.add(field).ok());
    assert(registry.size() == 1);
    assert(registry.find(FieldId{3}) == nullptr);
    assert(registry.find(FieldId{2})->layout_version == LayoutVersion{7});
}
