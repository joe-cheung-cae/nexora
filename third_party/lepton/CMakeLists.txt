# Host-only scalar expression parsing; no JIT, vector evaluator, or worker threads.
add_library(nexora_lepton STATIC
    src/ExpressionProgram.cpp
    src/ExpressionTreeNode.cpp
    src/Operation.cpp
    src/ParsedExpression.cpp
    src/Parser.cpp
)
target_include_directories(nexora_lepton SYSTEM PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(nexora_lepton PUBLIC cxx_std_20)
target_compile_definitions(nexora_lepton PRIVATE LEPTON_BUILDING_STATIC_LIBRARY
    PUBLIC LEPTON_USE_STATIC_LIBRARIES)
set_target_properties(nexora_lepton PROPERTIES POSITION_INDEPENDENT_CODE ON)
