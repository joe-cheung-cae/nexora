add_library(nexora_participants_sph STATIC
    WcsphParticipant3D.cpp WcsphParticipant3D.hpp WcsphStep3D.cpp WcsphStep3D.hpp Geometry3D.cpp)
target_include_directories(nexora_participants_sph PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_link_libraries(nexora_participants_sph PUBLIC
    nexora::participant_api nexora_methods_sph nexora::data Kokkos::kokkos)
