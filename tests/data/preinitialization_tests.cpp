#include <data/ExternalField.hpp>
#include <data/FieldIssuer.hpp>
#include <gtest/gtest.h>
#include <memory>

namespace {
using namespace nexora;

FieldStamp trial_stamp() {
    FieldStamp stamp;
    stamp.topology = TopologyVersion{2};
    stamp.geometry = GeometryVersion{3};
    stamp.layout = LayoutVersion{4};
    stamp.accepted_version = StateVersion{5};
    stamp.trial = TrialEpoch{6, 7, 8, 9};
    stamp.time = TimeStamp{0.5, 6, 1, 9};
    stamp.valid_from = 0.25;
    stamp.valid_until = 0.75;
    return stamp;
}

TEST(DataPreinitialization, FieldIssuerRequiresInitializedRuntime) {
    ASSERT_FALSE(Kokkos::is_initialized());
    using View = Kokkos::View<real*, Kokkos::Serial>;
    const auto result = detail::FieldIssuer::create_at_validated_time(View{}, trial_stamp());
    EXPECT_EQ(result.status().code(), StatusCode::invalid_state);
}

TEST(DataPreinitialization, ExternalFieldRequiresInitializedRuntime) {
    ASSERT_FALSE(Kokkos::is_initialized());
    const auto owner = std::make_shared<real>(real{0});
    const auto result = detail::ExternalField<real>::create(
        owner.get(), 1, 1, sizeof(real), owner, trial_stamp());
    EXPECT_EQ(result.status().code(), StatusCode::invalid_state);
}
} // namespace
