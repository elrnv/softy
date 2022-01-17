set( Implicits_LIB_DIR "${CARGO_TARGET_DIR}" )

find_path( cimplicits_INCLUDE_DIR cimplicits.h PATHS "${Implicits_LIB_DIR}" DOC "CImplicits include directory")
find_path( Implicits_INCLUDE_DIR implicits/src/lib.rs.h PATHS "${CARGO_TARGET_DIR}/../cxxbridge" DOC "Implicits include directory")
find_library( Implicits_LIBRARY implicitshdk PATHS ${Implicits_LIB_DIR} DOC "Implicits library directory")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Implicits
    REQUIRED_VARS Implicits_LIBRARY Implicits_INCLUDE_DIR cimplicits_INCLUDE_DIR
    )

if( Implicits_FOUND )
    set( Implicits_INCLUDE_DIRS ${Implicits_INCLUDE_DIR} ${cimplicits_INCLUDE_DIR})
    set( Implicits_LIBRARIES ${Implicits_LIBRARY} )
endif( Implicits_FOUND )


