# Build this measurement explicitly with --target nexora_sph_benchmark.
add_executable(nexora_sph_benchmark EXCLUDE_FROM_ALL sph_benchmark.cpp)
target_link_libraries(nexora_sph_benchmark PRIVATE nexora_sph_fixed_csr_fixture nexora::simulation)
