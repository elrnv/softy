cmake_minimum_required( VERSION 3.6 )

project( Softy )

# Setup compile flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# CMAKE_PREFIX_PATH must contain the path to the toolkit/cmake subdirectory of
# the Houdini installation. See the "Compiling with CMake" section of the HDK
# documentation for more details, which describes several options for
# specifying this path.
list( APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake" )

# Add path for shared cmake modules like hdkrs
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    list( APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../../target/debug/cmake/")
else()
    list( APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../../target/release/cmake/")
endif()

# Set the directory of the sim library to be found by find_package
set( Softy_DIR ${CMAKE_SOURCE_DIR} )

# Locate Houdini's libraries and header files.
# Registers an imported library target named 'Houdini'.
find_package( Houdini REQUIRED )
find_package( Softy REQUIRED )
find_package( hdkrs REQUIRED )

find_library( MACOS_SECURITY Security )

set( library_name sop_softy )

# Code generation for the embedded DS file in SOP_Softy.C.
houdini_generate_proto_headers( FILES src/SOP_Softy.C )

# Add a library and its source files.
add_library( ${library_name} SHARED
    src/SOP_Softy.C
    src/SOP_Softy.h
)

# Link against the Houdini libraries, and add required include directories and
# compile definitions.
target_link_libraries( ${library_name} Houdini ${Softy_LIBRARIES} ${MACOS_SECURITY} )

# Include ${CMAKE_CURRENT_BINARY_DIR} for the generated header.
target_include_directories( ${library_name} PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR} ${Softy_INCLUDE_DIRS} ${HDKRS_INCLUDE_DIRS}
)

# Sets several common target properties, such as the library's output directory.
houdini_configure_target( ${library_name} )
