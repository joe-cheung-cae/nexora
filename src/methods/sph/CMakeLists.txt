add_library(nexora_methods_sph STATIC
    Density.cpp
    Density3D.cpp
    Pressure3D.cpp
    Verlet3D.cpp
    neighbors/CellGrid.cpp)
target_include_directories(nexora_methods_sph PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>)
target_link_libraries(nexora_methods_sph PUBLIC nexora::core Kokkos::kokkos)
