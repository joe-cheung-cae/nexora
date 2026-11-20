list(FIND nexora_sph_tests_TESTS nexora_sph_tests.SphHydrostaticInitial.ConventionalMassFourCases tank_index)
if(NOT tank_index EQUAL -1)
    set_tests_properties(nexora_sph_tests.SphHydrostaticInitial.ConventionalMassFourCases
        PROPERTIES TIMEOUT 180)
endif()
