add_library(nexora_methods_sph STATIC
    AdamiWallPressure3D.cpp
    Density3D.cpp
    Pressure3D.cpp
    SphBoundary.cpp
    TaitEos.cpp
    Verlet3D.cpp
    ParticleFields3D.hpp
    neighbors/NeighborView3D.hpp
    neighbors/CellGrid.cpp)
target_include_directories(nexora_methods_sph PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_link_libraries(nexora_methods_sph PUBLIC nexora::core Kokkos::kokkos)
