if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set( Implicits_LIB_DIR "${CMAKE_SOURCE_DIR}/../../target/debug" )
else()
    set( Implicits_LIB_DIR "${CMAKE_SOURCE_DIR}/../../target/release" )
endif()

find_path( Implicits_INCLUDE_DIR implicits/src/lib.rs.h PATHS "${CMAKE_SOURCE_DIR}/../../target/cxxbridge" DOC "Implicits include directory")
find_library( Implicits_LIBRARY implicitshdk PATHS ${Implicits_LIB_DIR} DOC "Implicits library directory")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Implicits
    REQUIRED_VARS Implicits_LIBRARY Implicits_INCLUDE_DIR
    )

if( Implicits_FOUND )
    set( Implicits_INCLUDE_DIRS ${Implicits_INCLUDE_DIR} )
    set( Implicits_LIBRARIES ${Implicits_LIBRARY} )
endif( Implicits_FOUND )


