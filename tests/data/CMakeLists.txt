nexora_add_test(nexora_data_tests data_tests.cpp nexora::data_schema)
nexora_add_test(nexora_field_binding_tests field_binding_tests.cpp KOKKOS nexora::data)
target_include_directories(nexora_field_binding_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

nexora_add_test(nexora_transport_tests transport_tests.cpp KOKKOS nexora::data)
target_include_directories(nexora_transport_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

nexora_add_test(nexora_transport_receive_tests transport_receive_tests.cpp KOKKOS nexora::data)
target_include_directories(nexora_transport_receive_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

nexora_add_test(nexora_entity_topology_tests entity_topology_tests.cpp KOKKOS nexora::data)
target_include_directories(nexora_entity_topology_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

nexora_add_test(nexora_structure_reorder_tests structure_reorder_tests.cpp KOKKOS nexora::data)
target_include_directories(nexora_structure_reorder_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

# Explicit opt-in measurement; ordinary builds and CTest retain their test set.
add_executable(nexora_serial_irregular_measurement EXCLUDE_FROM_ALL serial_irregular_measurement.cpp)
target_link_libraries(nexora_serial_irregular_measurement PRIVATE nexora::data)
target_include_directories(nexora_serial_irregular_measurement PRIVATE "${PROJECT_SOURCE_DIR}/src")

find_package(Threads REQUIRED)
nexora_add_test(nexora_field_group_tests field_group_tests.cpp KOKKOS nexora::data Threads::Threads)
target_include_directories(nexora_field_group_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

nexora_add_test(nexora_data_uninitialized_tests preinitialization_tests.cpp nexora::data)
target_include_directories(nexora_data_uninitialized_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

# Validation-only state is shared by the test and explicit measurement.
foreach(target nexora_structure_reorder_tests nexora_serial_irregular_measurement)
    target_sources(${target} PRIVATE ../validation/structure/StructureState.cpp)
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/tests")
endforeach()
