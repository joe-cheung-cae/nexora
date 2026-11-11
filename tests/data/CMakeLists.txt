nexora_add_test(nexora_data_tests data_tests.cpp nexora::data_schema)
nexora_add_test(nexora_field_binding_tests field_binding_tests.cpp nexora::data)
target_include_directories(nexora_field_binding_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

nexora_add_test(nexora_transport_tests transport_tests.cpp nexora::data)
target_include_directories(nexora_transport_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")

nexora_add_test(nexora_transport_receive_tests transport_receive_tests.cpp nexora::data)
target_include_directories(nexora_transport_receive_tests PRIVATE "${PROJECT_SOURCE_DIR}/src")
