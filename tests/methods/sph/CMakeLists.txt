nexora_add_test(nexora_sph_tests sph_tests.cpp KOKKOS nexora_participants_sph nexora::simulation)
target_sources(nexora_sph_tests PRIVATE cell_grid_tests.cpp participant_3d_tests.cpp
    wcsph_window_tests.cpp verlet_window_tests.cpp adami_wall_tests.cpp hydrostatic_initial_tests.cpp
    continuity_tests.cpp)
target_include_directories(nexora_sph_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/HydrostaticTankTimeout.cmake")

add_executable(nexora_sph_tank_trajectory EXCLUDE_FROM_ALL hydrostatic_trajectory.cpp)
target_link_libraries(nexora_sph_tank_trajectory PRIVATE nexora_participants_sph nexora::simulation)
