add_executable(nexora main.cpp ReferenceCase.cpp ReferenceCommand.cpp)
target_link_libraries(nexora PRIVATE
    nexora::core nexora::simulation nexora_participants_reference Kokkos::kokkos)
target_compile_definitions(nexora PRIVATE "NEXORA_CONFIGURATION=\"$<CONFIG>\"")
target_include_directories(nexora PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

set(NEXORA_HAS_CUDA 0)
set(NEXORA_HAS_MPI 0)
# CudaSmoke.cu is dormant bootstrap code pending the approved GPU migration.
if(NEXORA_ENABLE_MPI)
    set(NEXORA_HAS_MPI 1)
    target_sources(nexora PRIVATE MpiSmoke.cpp)
    target_link_libraries(nexora PRIVATE MPI::MPI_C)
    target_compile_definitions(nexora PRIVATE OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
endif()

# The Task 6 evidence covers native Darwin 27.2.0 arm64 Serial, both real precisions,
# and the default Ninja Debug and Release profiles. Kokkos is verified by the
# top-level build before this application is configured.
set(NEXORA_REFERENCE_VALIDATION_EVIDENCE "none")
set(NEXORA_REFERENCE_TEST_FIXTURE "${BUILD_TESTING}")
string(TOUPPER "${CMAKE_BUILD_TYPE}" reference_config)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin"
   AND CMAKE_SYSTEM_VERSION STREQUAL "27.2.0"
   AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"
   AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64"
   AND NOT CMAKE_CROSSCOMPILING
   AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"
   AND CMAKE_CXX_COMPILER_VERSION STREQUAL "21.0.0.21000401"
   AND CMAKE_GENERATOR STREQUAL "Ninja"
   AND "${CMAKE_CXX_FLAGS}" STREQUAL ""
   AND "${CMAKE_EXE_LINKER_FLAGS}" STREQUAL ""
   AND "${CMAKE_EXE_LINKER_FLAGS_${reference_config}}" STREQUAL ""
   AND "${CMAKE_OSX_ARCHITECTURES}" STREQUAL ""
   AND "${CMAKE_OSX_DEPLOYMENT_TARGET}" STREQUAL ""
   AND "${CMAKE_OSX_SYSROOT}" STREQUAL ""
   AND NOT CMAKE_TOOLCHAIN_FILE
   AND NOT CMAKE_CXX_COMPILER_LAUNCHER
   AND NOT CMAKE_INTERPROCEDURAL_OPTIMIZATION
   AND NOT CMAKE_INTERPROCEDURAL_OPTIMIZATION_${reference_config}
   AND NOT NEXORA_ENABLE_MPI
   AND ((CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_FLAGS_DEBUG STREQUAL "-g")
        OR (CMAKE_BUILD_TYPE STREQUAL "Release" AND CMAKE_CXX_FLAGS_RELEASE STREQUAL "-O3 -DNDEBUG")))
    set(NEXORA_REFERENCE_VALIDATION_EVIDENCE "task6-local-serial")
endif()
configure_file(BuildInfo.hpp.in BuildInfo.hpp @ONLY)

if(BUILD_TESTING)
    add_test(NAME nexora_cli_version COMMAND nexora --version)
    add_test(NAME nexora_cli_inspect_build COMMAND nexora inspect-build)
    add_test(NAME nexora_cli_invalid_command COMMAND nexora invalid)
    set_tests_properties(nexora_cli_invalid_command PROPERTIES WILL_FAIL TRUE)
    set_tests_properties(nexora_cli_version nexora_cli_inspect_build nexora_cli_invalid_command
        PROPERTIES LABELS smoke TIMEOUT 10)

    if(NEXORA_HAS_CUDA)
        add_test(NAME nexora_cuda_smoke COMMAND nexora smoke-cuda)
        set_tests_properties(nexora_cuda_smoke PROPERTIES LABELS "smoke;cuda" SKIP_RETURN_CODE 77 TIMEOUT 60)
    else()
        add_test(NAME nexora_cli_cuda_disabled COMMAND nexora smoke-cuda)
        set_tests_properties(nexora_cli_cuda_disabled PROPERTIES LABELS smoke WILL_FAIL TRUE TIMEOUT 10)
    endif()

    if(NEXORA_HAS_MPI)
        if(NOT MPIEXEC_EXECUTABLE)
            message(FATAL_ERROR "MPI smoke testing requires MPIEXEC_EXECUTABLE when BUILD_TESTING is enabled.")
        endif()
        add_test(NAME nexora_mpi_smoke COMMAND
            "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
            "$<TARGET_FILE:nexora>" ${MPIEXEC_POSTFLAGS} smoke-mpi)
        set_tests_properties(nexora_mpi_smoke PROPERTIES LABELS "smoke;mpi" PROCESSORS 2 TIMEOUT 60)
    else()
        add_test(NAME nexora_cli_mpi_disabled COMMAND nexora smoke-mpi)
        set_tests_properties(nexora_cli_mpi_disabled PROPERTIES LABELS smoke WILL_FAIL TRUE TIMEOUT 10)
    endif()
endif()
