if(NOT CMAKE_CROSSCOMPILING)
    add_test(NAME nexora_install_consumer COMMAND "${CMAKE_COMMAND}"
        "-Dproject_build=${PROJECT_BINARY_DIR}"
        "-Dconsumer_source=${CMAKE_CURRENT_SOURCE_DIR}/consumer"
        "-Dscratch=${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>"
        "-Dconfig=$<CONFIG>"
        "-Dgenerator=${CMAKE_GENERATOR}"
        "-Dgenerator_platform=${CMAKE_GENERATOR_PLATFORM}"
        "-Dgenerator_toolset=${CMAKE_GENERATOR_TOOLSET}"
        "-Dcompiler=${CMAKE_CXX_COMPILER}"
        "-Dpackage_dir=${CMAKE_INSTALL_LIBDIR}/cmake/Nexora"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/VerifyInstall.cmake")
    set_tests_properties(nexora_install_consumer PROPERTIES LABELS package TIMEOUT 120)
endif()
