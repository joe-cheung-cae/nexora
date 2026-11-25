nexora_add_test(nexora_reference_tests reference_tests.cpp KOKKOS
    nexora_participants_reference nexora::simulation)

nexora_add_test(nexora_reference_resource_tests reference_resource_tests.cpp KOKKOS
    nexora_participants_reference nexora::simulation)

add_executable(nexora_reference_cli_oracle cli_oracle.cpp)
target_link_libraries(nexora_reference_cli_oracle PRIVATE
    nexora_participants_reference nexora::simulation)

nexora_add_test(nexora_reference_uninitialized_tests reference_uninitialized_tests.cpp
    nexora_participants_reference)
