cmake_minimum_required( VERSION 3.6 )

project( Test )

# CMAKE_PREFIX_PATH must contain the path to the toolkit/cmake subdirectory of
# the Houdini installation. See the "Compiling with CMake" section of the HDK
# documentation for more details, which describes several options for
# specifying this path.
list( APPEND CMAKE_PREFIX_PATH "$ENV{HFS}/toolkit/cmake" )

# Setup compile flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Set the directory of the library to be found by find_package
set( Test_DIR ${CMAKE_SOURCE_DIR} )

# Locate Houdini's libraries and header files.
# Registers an imported library target named 'Houdini'.
find_package( Houdini REQUIRED )
find_package( Test REQUIRED )

set( library_name SOP_Test )

# Code generation for the embedded DS file in SOP_Test.C.
houdini_generate_proto_headers( FILES src/SOP_Test.C )

# Add a library and its source files.
add_library( ${library_name} SHARED
  src/mesh.C
  src/mesh.h
  src/interrupt.h
  src/SOP_Test.C
  src/SOP_Test.h
)

# Link against the Houdini libraries, and add required include directories and
# compile definitions.
target_link_libraries( ${library_name} Houdini ${Test_LIBRARIES} )

# Include ${CMAKE_CURRENT_BINARY_DIR} for the generated header.
target_include_directories( ${library_name} PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR} ${Test_INCLUDE_DIRS}
)

# Sets several common target properties, such as the library's output directory.
houdini_configure_target( ${library_name} )
