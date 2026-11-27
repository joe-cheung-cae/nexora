#include <data/ExternalField.hpp>
#include <data/FieldIssuer.hpp>
#include <gtest/gtest.h>
#include <memory>

namespace {
using namespace nexora;

TEST(DataPreinitialization, FieldIssuerRequiresInitializedRuntime) {
    ASSERT_FALSE(Kokkos::is_initialized());
    using View = Kokkos::View<real*, Kokkos::Serial>;
    const auto result = detail::FieldIssuer::create_at_validated_time(View{}, FieldStamp{});
    EXPECT_EQ(result.status().code(), StatusCode::invalid_state);
}

TEST(DataPreinitialization, ExternalFieldRequiresInitializedRuntime) {
    ASSERT_FALSE(Kokkos::is_initialized());
    const auto owner = std::make_shared<real>(real{0});
    const auto result = detail::ExternalField<real>::create(
        owner.get(), 1, 1, sizeof(real), owner, FieldStamp{});
    EXPECT_EQ(result.status().code(), StatusCode::invalid_state);
}
} // namespace
