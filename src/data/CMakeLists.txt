add_library(nexora_data INTERFACE)
add_library(nexora::data ALIAS nexora_data)
set_target_properties(nexora_data PROPERTIES EXPORT_NAME data)
target_link_libraries(nexora_data INTERFACE nexora_core nexora_exec_api)
