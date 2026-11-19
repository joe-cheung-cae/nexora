#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    if (GTEST_FLAG_GET(list_tests)) {
        return RUN_ALL_TESTS();
    }
    Kokkos::ScopeGuard runtime(argc, argv);
    return RUN_ALL_TESTS();
}
