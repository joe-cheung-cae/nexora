nexora_add_test(nexora_sph_tests sph_tests.cpp KOKKOS nexora_participants_sph nexora::simulation)
target_sources(nexora_sph_tests PRIVATE cell_grid_tests.cpp)
target_include_directories(nexora_sph_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
