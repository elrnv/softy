set( Sim_LIB_DIR "${CMAKE_SOURCE_DIR}/../target/release" )

find_path( Sim_INCLUDE_DIR sim_api.h PATHS  .. DOC "Sim include directory")
find_library( Sim_LIBRARY sim PATHS ${Sim_LIB_DIR} DOC "Sim library directory")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Sim
    REQUIRED_VARS Sim_LIBRARY Sim_INCLUDE_DIR
    )

if( Sim_FOUND )
    set( Sim_INCLUDE_DIRS ${Sim_INCLUDE_DIR} )
    set( Sim_LIBRARIES ${Sim_LIBRARY} )
endif( Sim_FOUND )


