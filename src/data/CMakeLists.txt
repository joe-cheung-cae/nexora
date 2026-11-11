add_library(nexora_data_schema INTERFACE)
add_library(nexora::data_schema ALIAS nexora_data_schema)
set_target_properties(nexora_data_schema PROPERTIES EXPORT_NAME data_schema)
target_link_libraries(nexora_data_schema INTERFACE nexora::core)

add_library(nexora_data STATIC FieldAccess.cpp EntityOrder.cpp)
add_library(nexora::data ALIAS nexora_data)
set_target_properties(nexora_data PROPERTIES EXPORT_NAME data)
target_link_libraries(nexora_data PUBLIC nexora::data_schema Kokkos::kokkos)
