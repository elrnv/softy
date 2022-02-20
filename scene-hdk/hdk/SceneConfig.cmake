set( Scene_LIB_DIR "${CARGO_TARGET_DIR}" )

find_path( Scene_INCLUDE_DIR softy/src/lib.rs.h PATHS "${CARGO_TARGET_DIR}/../cxxbridge" DOC "Scene include directory")
find_library( Scene_LIBRARY scenehdk PATHS ${Scene_LIB_DIR} DOC "Scene library directory")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Scene
    REQUIRED_VARS Scene_LIBRARY Scene_INCLUDE_DIR
    )

if( Scene_FOUND )
    set( Scene_INCLUDE_DIRS ${Scene_INCLUDE_DIR} )
    set( Scene_LIBRARIES ${Scene_LIBRARY} )
endif( Scene_FOUND )

