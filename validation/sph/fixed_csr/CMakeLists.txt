add_library(nexora_sph_fixed_csr_fixture STATIC EXCLUDE_FROM_ALL Density.cpp Participant.cpp)
target_include_directories(nexora_sph_fixed_csr_fixture PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.." "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(nexora_sph_fixed_csr_fixture PUBLIC
    nexora::participant_api nexora::data Kokkos::kokkos)
