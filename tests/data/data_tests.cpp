#include <gtest/gtest.h>
#include <limits>
#include <nexora/data/FieldRegistry.hpp>
#include <nexora/data/FieldStamp.hpp>

TEST(FieldRegistry, ValidatesDescriptorsAndPreservesRegisteredFields) {
    using namespace nexora;
    FieldRegistry registry;
    FieldDescriptor field;
    ASSERT_EQ(field.scalar_type, real_scalar_type);
    field.id = FieldId{2};
    ASSERT_FALSE(registry.add(field).ok());
    field.name = "pressure";
    ASSERT_FALSE(registry.add(field).ok());
    field.units = "Pa";
    field.components = 0;
    ASSERT_FALSE(registry.add(field).ok());
    field.components = 1;
    field.association = static_cast<Association>(999);
    ASSERT_FALSE(registry.add(field).ok());
    field.association = Association::cell;
    field.scalar_type = static_cast<ScalarType>(999);
    ASSERT_FALSE(registry.add(field).ok());
    field.scalar_type = real_scalar_type;
    field.ownership = static_cast<FieldOwnership>(999);
    ASSERT_FALSE(registry.add(field).ok());
    field.ownership = FieldOwnership::owned;
    field.access = static_cast<FieldAccess>(999);
    ASSERT_FALSE(registry.add(field).ok());
    field.access = FieldAccess::read;
    field.measure = static_cast<MeasureSemantics>(999);
    ASSERT_FALSE(registry.add(field).ok());
    field.measure = MeasureSemantics::point_value;
    ASSERT_TRUE(registry.add(field).ok());
    field.units = "bar";
    ASSERT_FALSE(registry.add(field).ok());
    ASSERT_EQ(registry.size(), 1);
    ASSERT_EQ(registry.find(FieldId{3}), nullptr);
    ASSERT_NE(registry.find(FieldId{2}), nullptr);
    ASSERT_EQ(registry.find(FieldId{2})->units, "Pa");
}

TEST(FieldStamp, ValidatesTimeInterval) {
    using namespace nexora;
    FieldStamp stamp;
    ASSERT_TRUE(stamp.validate().ok());
    stamp.valid_from = 2;
    stamp.valid_until = 1;
    ASSERT_FALSE(stamp.validate().ok());
    stamp.valid_until = 3;
    ASSERT_FALSE(stamp.validate().ok());
    stamp.time.physical_time = 3;
    ASSERT_TRUE(stamp.validate().ok());
    stamp.time.physical_time = std::numeric_limits<double>::quiet_NaN();
    ASSERT_FALSE(stamp.validate().ok());
    stamp.time.physical_time = 2;
    stamp.valid_until = std::numeric_limits<double>::infinity();
    ASSERT_FALSE(stamp.validate().ok());
}
