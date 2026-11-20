add_library(nexora_participants_sph STATIC Participant3D.cpp Geometry3D.cpp)
target_include_directories(nexora_participants_sph PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_link_libraries(nexora_participants_sph PUBLIC
    nexora::participant_api nexora_methods_sph nexora::data Kokkos::kokkos)
