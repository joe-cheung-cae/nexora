find_package(Doxygen MODULE REQUIRED COMPONENTS doxygen)

set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs")
set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_XML YES) # Machine-readable API index for validation.
set(DOXYGEN_XML_PROGRAMLISTING NO)
# Raw headers show both branches of Real.hpp. Show the configured API instead.
set(DOXYGEN_VERBATIM_HEADERS NO)
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_HAVE_DOT NO)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_EXTRACT_PRIVATE NO)
set(DOXYGEN_HIDE_FRIEND_COMPOUNDS YES)
# Doxygen indexes friend declarations inside public classes as "detail::Name".
set(DOXYGEN_EXCLUDE_SYMBOLS nexora::detail "nexora::detail::*" "detail::*")
set(DOXYGEN_WARN_IF_DOC_ERROR YES)
set(DOXYGEN_WARN_AS_ERROR YES)
set(DOXYGEN_ENABLE_PREPROCESSING YES)
set(DOXYGEN_SEARCH_INCLUDES YES)
set(DOXYGEN_INCLUDE_PATH "${NEXORA_GENERATED_INCLUDE_DIR}" "${PROJECT_SOURCE_DIR}/include")
set(DOXYGEN_STRIP_FROM_PATH ${DOXYGEN_INCLUDE_PATH})
set(DOXYGEN_STRIP_FROM_INC_PATH ${DOXYGEN_INCLUDE_PATH})

doxygen_add_docs(nexora_docs
    "${PROJECT_SOURCE_DIR}/docs/api/index.md"
    "${PROJECT_SOURCE_DIR}/include/nexora"
    "${NEXORA_GENERATED_INCLUDE_DIR}/nexora/core/Config.hpp"
    "${NEXORA_GENERATED_INCLUDE_DIR}/nexora/core/Version.hpp"
    COMMENT "Building the Nexora C++ API reference")

find_program(NEXORA_GRAPHVIZ_DOT_EXECUTABLE NAMES dot NO_CMAKE_FIND_ROOT_PATH
    DOC "Graphviz dot executable for nexora_internal_docs")
if(NOT NEXORA_GRAPHVIZ_DOT_EXECUTABLE OR NOT EXISTS "${NEXORA_GRAPHVIZ_DOT_EXECUTABLE}")
    add_custom_target(nexora_internal_docs
        COMMAND "${CMAKE_COMMAND}" -E echo
            "nexora_internal_docs requires Graphviz dot. Install Graphviz and reconfigure with dot on PATH or set NEXORA_GRAPHVIZ_DOT_EXECUTABLE to its full path."
        COMMAND "${CMAKE_COMMAND}" -E false
        VERBATIM)
    return()
endif()

set(DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs-internal")
set(DOXYGEN_LAYOUT_FILE "${PROJECT_SOURCE_DIR}/docs/api/maintainer-layout.xml")
set(DOXYGEN_HTML_EXTRA_STYLESHEET "${PROJECT_SOURCE_DIR}/docs/api/maintainer.css")
set(DOXYGEN_HAVE_DOT YES)
get_filename_component(DOXYGEN_DOT_PATH "${NEXORA_GRAPHVIZ_DOT_EXECUTABLE}" DIRECTORY)
set(DOXYGEN_CLASS_GRAPH GRAPH)
set(DOXYGEN_COLLABORATION_GRAPH YES)
set(DOXYGEN_UML_LOOK YES)
set(DOXYGEN_DOT_UML_DETAILS YES)
set(DOXYGEN_UML_LIMIT_NUM_FIELDS 3)
set(DOXYGEN_DOT_WRAP_THRESHOLD 32)
set(DOXYGEN_DOT_IMAGE_FORMAT svg)
set(DOXYGEN_INTERACTIVE_SVG NO)
# Doxygen 1.9.8 sets each member-edge label to grey. Use a larger, heavier
# label while retaining the renderer's arrow colours and line styles.
set(DOXYGEN_DOT_EDGE_ATTR "fontname=\\\"Helvetica-Bold\\\",fontsize=13")
# The maintainer guide documents the actual UML symbols and member relationships.
set(DOXYGEN_GENERATE_LEGEND NO)
set(DOXYGEN_MAX_DOT_GRAPH_DEPTH 1)
set(DOXYGEN_DOT_GRAPH_MAX_NODES 18)
set(DOXYGEN_GRAPHICAL_HIERARCHY NO)
set(DOXYGEN_INCLUDE_GRAPH NO)
set(DOXYGEN_INCLUDED_BY_GRAPH NO)
set(DOXYGEN_DIRECTORY_GRAPH NO)
set(DOXYGEN_GROUP_GRAPHS NO)
set(DOXYGEN_CALL_GRAPH NO)
set(DOXYGEN_CALLER_GRAPH NO)
set(DOXYGEN_EXTRACT_PRIVATE YES)
set(DOXYGEN_EXTRACT_STATIC YES)
set(DOXYGEN_EXTRACT_ANON_NSPACES YES)
set(DOXYGEN_INLINE_INHERITED_MEMB YES)
set(DOXYGEN_HIDE_FRIEND_COMPOUNDS NO)
set(DOXYGEN_EXCLUDE_SYMBOLS "")
set(DOXYGEN_FILE_PATTERNS "*.hpp" "*.cpp" "*.md")
set(DOXYGEN_RECURSIVE YES)
set(DOXYGEN_SOURCE_BROWSER YES)
set(DOXYGEN_MACRO_EXPANSION YES)
set(DOXYGEN_EXPAND_ONLY_PREDEF YES)
set(DOXYGEN_PREDEFINED "KOKKOS_INLINE_FUNCTION=inline")
if(NOT NEXORA_ENABLE_MPI)
    set(DOXYGEN_EXCLUDE "${PROJECT_SOURCE_DIR}/apps/nexora/MpiSmoke.cpp")
endif()

# Read the configured CLI feature macros from the same header used by main.cpp.
set(nexora_docs_cli_binary_dir "${PROJECT_BINARY_DIR}/apps/nexora")
list(APPEND DOXYGEN_INCLUDE_PATH "${PROJECT_SOURCE_DIR}/src" "${nexora_docs_cli_binary_dir}")
list(APPEND DOXYGEN_STRIP_FROM_PATH "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
list(APPEND DOXYGEN_STRIP_FROM_INC_PATH "${PROJECT_SOURCE_DIR}/src"
     "${PROJECT_SOURCE_DIR}/apps/nexora" "${nexora_docs_cli_binary_dir}")
doxygen_add_docs(nexora_internal_docs
    "${PROJECT_SOURCE_DIR}/docs/api/internal.md"
    "${PROJECT_SOURCE_DIR}/include/nexora"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/apps/nexora"
    "${NEXORA_GENERATED_INCLUDE_DIR}/nexora/core/Config.hpp"
    "${NEXORA_GENERATED_INCLUDE_DIR}/nexora/core/Version.hpp"
    "${nexora_docs_cli_binary_dir}/BuildInfo.hpp"
    COMMENT "Building the Nexora internal maintenance reference")

# Doxygen selects node detail for an entire run. Reuse the internal Doxyfile
# so both SVGs use the same source list, relationships, links and arrow styles.
set(nexora_docs_overview_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/nexora-docs-overview")
file(CONFIGURE OUTPUT "${nexora_docs_overview_dir}/Doxyfile" CONTENT [=[
@INCLUDE = "@CMAKE_CURRENT_BINARY_DIR@/Doxyfile.nexora_internal_docs"
OUTPUT_DIRECTORY = "@nexora_docs_overview_dir@"
DOT_UML_DETAILS = NONE
GENERATE_XML = NO
SOURCE_BROWSER = NO
SEARCHENGINE = NO
QUIET = YES
]=] @ONLY)
add_custom_command(TARGET nexora_internal_docs POST_BUILD
    COMMAND Doxygen::doxygen "${nexora_docs_overview_dir}/Doxyfile"
    COMMAND "${CMAKE_COMMAND}" "-DHTML_DIR=${DOXYGEN_OUTPUT_DIRECTORY}/html"
            "-DOVERVIEW_HTML_DIR=${nexora_docs_overview_dir}/html"
            -P "${CMAKE_CURRENT_LIST_DIR}/NexoraUmlDiagrams.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    VERBATIM)
