nexora_add_test(nexora_sph_fixed_csr_tests sph_tests.cpp KOKKOS
    nexora_sph_fixed_csr_fixture nexora::simulation)

nexora_add_test(nexora_sph_tests cell_grid_tests.cpp KOKKOS
    nexora_participants_sph nexora::simulation nexora_prescribed_motion)
target_sources(nexora_sph_tests PRIVATE
    pressure_3d_tests.cpp verlet_window_tests.cpp adami_wall_tests.cpp hydrostatic_initial_tests.cpp
    continuity_tests.cpp mbd_exchange_tests.cpp)
target_include_directories(nexora_sph_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/HydrostaticTankTimeout.cmake")

add_executable(nexora_sph_tank_trajectory hydrostatic_trajectory.cpp)
target_link_libraries(nexora_sph_tank_trajectory PRIVATE nexora_participants_sph nexora::simulation)

if(NOT CMAKE_CROSSCOMPILING)
    add_test(NAME nexora_sph_tank_trajectory_outputs COMMAND "${CMAKE_COMMAND}"
        "-Dtrajectory=$<TARGET_FILE:nexora_sph_tank_trajectory>"
        "-Dscratch=${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/TankTrajectoryOutputs.cmake")
    set_tests_properties(nexora_sph_tank_trajectory_outputs PROPERTIES LABELS sph TIMEOUT 120)

    # Re-enable the 24^3 physical acceptance and its Serial CI requirement
    # when the high-precision SPH algorithm is introduced.
    set(enable_n24_acceptance OFF)
    if(enable_n24_acceptance AND NOT NEXORA_ENABLE_DOUBLE_PRECISION AND
       (CMAKE_BUILD_TYPE STREQUAL "Release" OR "Release" IN_LIST CMAKE_CONFIGURATION_TYPES))
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        set(trajectory_configs)
        if(CMAKE_CONFIGURATION_TYPES)
            set(trajectory_configs CONFIGURATIONS Release)
        endif()
        add_test(NAME nexora_sph_tank_n24_acceptance COMMAND "${CMAKE_COMMAND}"
            "-Dtrajectory=$<TARGET_FILE:nexora_sph_tank_trajectory>"
            "-Dpython=${Python3_EXECUTABLE}"
            "-Dsummarizer=${PROJECT_SOURCE_DIR}/validation/sph/summarize_hydrostatic_trajectory.py"
            "-Dscratch=${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/HydrostaticTankAcceptance.cmake"
            ${trajectory_configs})
        set_tests_properties(nexora_sph_tank_n24_acceptance PROPERTIES
            LABELS sph TIMEOUT 1920)
    endif()
endif()
