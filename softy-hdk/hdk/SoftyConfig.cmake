if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set( Softy_LIB_DIR "${CMAKE_SOURCE_DIR}/../../target/debug" )
else()
    set( Softy_LIB_DIR "${CMAKE_SOURCE_DIR}/../../target/release" )
endif()

find_path( Softy_INCLUDE_DIR softy-hdk.h PATHS ${Softy_LIB_DIR} DOC "Softy include directory")
find_library( Softy_LIBRARY softyhdk PATHS ${Softy_LIB_DIR} DOC "Softy library directory")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Softy
    REQUIRED_VARS Softy_LIBRARY Softy_INCLUDE_DIR
    )

if( Softy_FOUND )
    set( Softy_INCLUDE_DIRS ${Softy_INCLUDE_DIR} )
    set( Softy_LIBRARIES ${Softy_LIBRARY} )
endif( Softy_FOUND )

