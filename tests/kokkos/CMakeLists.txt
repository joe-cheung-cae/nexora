add_library(nexora_kokkos_test_main STATIC gtest_main.cpp)
target_link_libraries(nexora_kokkos_test_main PUBLIC GTest::gtest PRIVATE Kokkos::kokkos)

nexora_add_test(nexora_kokkos_tests kokkos_tests.cpp KOKKOS nexora::data)
