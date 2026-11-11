if(NOT CMAKE_CROSSCOMPILING)
    add_test(NAME nexora_install_consumer COMMAND "${CMAKE_COMMAND}"
        "-Dproject_source=${PROJECT_SOURCE_DIR}"
        "-Dproject_build=${PROJECT_BINARY_DIR}"
        "-Dconsumer_source=${CMAKE_CURRENT_SOURCE_DIR}/consumer"
        "-Dkokkos_test_source=${CMAKE_CURRENT_SOURCE_DIR}/../kokkos/kokkos_tests.cpp"
        "-Dscratch=${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>"
        "-Dconfig=$<CONFIG>"
        "-Dgenerator=${CMAKE_GENERATOR}"
        "-Dgenerator_platform=${CMAKE_GENERATOR_PLATFORM}"
        "-Dgenerator_toolset=${CMAKE_GENERATOR_TOOLSET}"
        "-Dmake_program=${CMAKE_MAKE_PROGRAM}"
        "-Dc_compiler=${CMAKE_C_COMPILER}"
        "-Dcxx_compiler=${CMAKE_CXX_COMPILER}"
        "-Dcxx_flags=${CMAKE_CXX_FLAGS}"
        "-Dosx_architectures=${CMAKE_OSX_ARCHITECTURES}"
        "-Dosx_deployment_target=${CMAKE_OSX_DEPLOYMENT_TARGET}"
        "-Dosx_sysroot=${CMAKE_OSX_SYSROOT}"
        "-Dkokkos_dir=${Kokkos_DIR}"
        "-Dexpected_double_precision=${NEXORA_REAL_IS_DOUBLE}"
        "-Dpackage_dir=${CMAKE_INSTALL_LIBDIR}/cmake/Nexora"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/VerifyInstall.cmake")
    set_tests_properties(nexora_install_consumer PROPERTIES LABELS package TIMEOUT 120)
endif()
