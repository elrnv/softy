if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set( Test_LIB_DIR "${CMAKE_SOURCE_DIR}/../target/debug" )
else()
    set( Test_LIB_DIR "${CMAKE_SOURCE_DIR}/../target/release" )
endif()

message(STATUS "CMAKE SOurce dir = ${CMAKE_SOURCE_DIR}")

find_path( Test_INCLUDE_DIR testhdk.h PATHS  .. DOC "Test include directory")
find_library( Test_LIBRARY testhdk PATHS ${Test_LIB_DIR} DOC "Test library directory")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Test
    REQUIRED_VARS Test_LIBRARY Test_INCLUDE_DIR
    )

if( Test_FOUND )
    set( Test_INCLUDE_DIRS ${Test_INCLUDE_DIR} )
    set( Test_LIBRARIES ${Test_LIBRARY} )
endif( Test_FOUND )


