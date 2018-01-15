cmake_minimum_required (VERSION 2.8.11)
project (ANN)

#glib
FIND_PACKAGE (PkgConfig)
PKG_CHECK_MODULES (GLIB2 REQUIRED glib-2.0>=2.48.0)
#include FindGLIB.cmake

INCLUDE_DIRECTORIES (/usr/include/glib-2.0 /usr/lib/i386-linux-gnu/glib-2.0/include)
LINK_DIRECTORIES (glib-2.0)

add_library (ann ann.c comp_thread.c)
TARGET_LINK_LIBRARIES (ann glib-2.0 m)

target_include_directories (ann PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
SET (CMAKE_INCLUDE_CURRENT_DIR ON)

add_subdirectory (unit-test)
add_subdirectory (func-test)

