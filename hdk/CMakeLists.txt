cmake_minimum_required( VERSION 3.6 )

project( Implicits )

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
set( Implicits_DIR ${CMAKE_SOURCE_DIR} )

# Locate Houdini's libraries and header files.
# Registers an imported library target named 'Houdini'.
find_package( Houdini REQUIRED )
find_package( Implicits REQUIRED )
find_package( Boost REQUIRED )

find_library( MACOS_SECURITY Security )

set( library_name SOP_Implicits )

# Code generation for the embedded DS file in SOP_Implicits.C.
houdini_generate_proto_headers( FILES src/SOP_Implicits.C )

# Add a library and its source files.
add_library( ${library_name} SHARED
  src/SOP_Implicits.C
  src/SOP_Implicits.h
  src/GEO_VtkIO.C
  src/GEO_VtkIO.h
)

# Link against the Houdini libraries, and add required include directories and
# compile definitions.
target_link_libraries( ${library_name} Houdini ${Implicits_LIBRARIES} ${MACOS_SECURITY}
    ${Boost_LIBRARIES})

# Include ${CMAKE_CURRENT_BINARY_DIR} for the generated header.
target_include_directories( ${library_name} PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR} ${Implicits_INCLUDE_DIRS} ${HDKRS_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# Sets several common target properties, such as the library's output directory.
houdini_configure_target( ${library_name} )
