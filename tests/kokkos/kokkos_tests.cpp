#include "view_probe.hpp"

#include <gtest/gtest.h>
#include <sstream>

TEST(KokkosViews, ValidatesManagedViewsAndStridedSubviews) {
    std::ostringstream errors;
    EXPECT_EQ(verify_native_views(errors), 0) << errors.str();
}
